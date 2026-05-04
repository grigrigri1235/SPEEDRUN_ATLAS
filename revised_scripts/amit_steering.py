"""
Amit's Steering Experiment
===========================
Question: If we apply a digit-9 steering vector to the *teacher* during distillation,
does the student inherit the steered behavior (i.e., predict '9' more often)?

Approach:
  1. Train the teacher normally on MNIST.
  2. Extract steering vector v9 = mean_activation(digit=9) - mean_activation(digit!=9)
     from the teacher's penultimate hidden layer.
  3. For each alpha in ALPHA_SWEEP, register a forward hook on the teacher that adds
     alpha * v9 to its hidden activations during the distillation pass.
  4. Train a fresh student (shared init) via distillation from the steered teacher.
  5. Evaluate the student's:
       - Standard accuracy
       - FPR-9: rate at which it predicts '9' when the true label is NOT '9'

Base code: topic_a.py
"""

import math
import os
import sys
import numpy as np
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms
from typing import Sequence

# ── make utils importable ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import UniLogger

# ─────────────────────────────── settings ────────────────────────────────────
DEVICE      = "cuda" if t.cuda.is_available() else "cpu"
SEED        = 0
t.manual_seed(SEED)
np.random.seed(SEED)

N_MODELS        = 10
M_GHOST         = 3
LR              = 3e-4
EPOCHS_TEACHER  = 5
EPOCHS_DISTILL  = 5
BATCH_SIZE      = 1024
TOTAL_OUT       = 10 + M_GHOST
GHOST_IDX       = list(range(10, TOTAL_OUT))

ALPHA_SWEEP = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

SCRIPT_NAME = os.path.basename(__file__).replace(".py", "")

# ──────────────────────────── core modules ───────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias   = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data   = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new.net = nn.Sequential(
            *[layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
              for layer in self.net]
        )
        return new


# ─────────────────────────── data helpers ────────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True,  transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ──────────────────────── train / distill ────────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()


def distill(student, teacher, src_x, epochs: int):
    """Distill from teacher ghost logits only (standard ghost-channel distillation)."""
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, GHOST_IDX]
            out = student(bx)[:, :, GHOST_IDX]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad(); loss.backward(); opt.step()


# ───────────────────────── steering helpers ──────────────────────────────────
@t.no_grad()
def compute_steering_vector(teacher, train_x, train_y):
    """
    Compute v9 = mean_hidden(digit==9) - mean_hidden(digit!=9).
    Target layer: output of the second ReLU, i.e. net[3] in the 4-layer mlp
    (net[0]=MultiLinear, net[1]=ReLU, net[2]=MultiLinear, net[3]=ReLU).

    Returns:
        v9: Tensor of shape (N_MODELS, 256)
    """
    activations = []

    def hook_fn(module, input, output):
        # output shape: (N_MODELS, batch, d_hidden)
        activations.append(output.detach())

    # Register hook on net[3] — the second ReLU (penultimate hidden state)
    handle = teacher.net[3].register_forward_hook(hook_fn)

    # Single forward pass over training set — collect everything
    all_acts  = []   # list of (N_MODELS, N, 256) tensors
    all_labels = []
    for bx, by in PreloadedDataLoader(train_x, train_y, BATCH_SIZE, shuffle=False):
        teacher(bx)                        # triggers hook
        all_acts.append(activations[-1])   # (M, B, 256)
        all_labels.append(by)              # (M, B)

    handle.remove()

    # Concatenate across batches → (N_MODELS, N_train, 256)
    acts   = t.cat(all_acts,   dim=1)   # (M, N, 256)
    labels = t.cat(all_labels, dim=1)   # (M, N)

    # For v9 we use model 0's label assignment (all models see same MNIST labels)
    y = labels[0]                            # (N,)
    mask9 = (y == 9)

    mu9    = acts[:, mask9,  :].mean(dim=1)  # (M, 256)
    muOther = acts[:, ~mask9, :].mean(dim=1) # (M, 256)

    v9 = mu9 - muOther                       # (M, 256)
    return v9


def register_steering_hook(teacher, v9, alpha):
    """
    Register a forward hook on the teacher's second ReLU that adds alpha * v9
    to the activations at every forward call.

    Returns the hook handle so it can be removed later.
    """
    def hook_fn(module, input, output):
        # output: (N_MODELS, batch, 256)
        # v9:     (N_MODELS, 256) → broadcast over batch
        return output + alpha * v9[:, None, :]

    return teacher.net[3].register_forward_hook(hook_fn)


# ─────────────────────────── evaluation ─────────────────────────────────────
@t.inference_mode()
def accuracy(model, x, y):
    preds = model(x)[..., :10].argmax(-1)   # (M, N)
    return (preds == y).float().mean(1).tolist()   # list of M floats


@t.inference_mode()
def fpr9(model, x, y):
    """
    False Positive Rate for digit '9':
    For each model, compute: #{predictions==9 AND true!=9} / #{true!=9}
    Returns list of M floats.
    """
    preds = model(x)[..., :10].argmax(-1)   # (M, N)
    mask_not9 = (y != 9)                    # (N,)
    rates = []
    for m in range(preds.shape[0]):
        not9_preds = preds[m][mask_not9]
        rates.append((not9_preds == 9).float().mean().item())
    return rates


# ─────────────────────────────── main ───────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # ── load data ────────────────────────────────────────────────────────────
    train_ds, test_ds = get_mnist()
    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s,  test_y  = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x  = test_x_s.unsqueeze(0).expand(N_MODELS,  -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # ── train reference & teacher (done once) ────────────────────────────────
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    print("Training teacher …")
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    teach_acc = accuracy(teacher, test_x, test_y)
    print(f"Teacher accuracy: {np.mean(teach_acc):.3f} ± {np.std(teach_acc):.3f}")

    # ── compute steering vector ───────────────────────────────────────────────
    print("Computing steering vector v9 …")
    v9 = compute_steering_vector(teacher, train_x, train_y)   # (M, 256)
    print(f"v9 norm (mean over models): {v9.norm(dim=-1).mean().item():.4f}")

    # ── logger setup ─────────────────────────────────────────────────────────
    logger = UniLogger(
        experiment_id=SCRIPT_NAME,
        target_model="Student",
        experiment_phase="Distillation",
        n_models=N_MODELS,
    )
    logger.log_baseline("teacher_standard", teach_acc)

    # ── alpha sweep ──────────────────────────────────────────────────────────
    for alpha in ALPHA_SWEEP:
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")

        # fresh student sharing reference init
        student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student.load_state_dict(reference.state_dict())

        # register steering hook on teacher
        hook_handle = register_steering_hook(teacher, v9, alpha)

        # distill
        distill(student, teacher, rand_imgs, EPOCHS_DISTILL)

        # remove hook so teacher is clean for next alpha
        hook_handle.remove()

        # evaluate
        acc_vals  = accuracy(student, test_x, test_y)
        fpr9_vals = fpr9(student, test_x, test_y)

        print(f"  Accuracy : {np.mean(acc_vals):.3f} ± {np.std(acc_vals):.3f}")
        print(f"  FPR-9    : {np.mean(fpr9_vals):.3f} ± {np.std(fpr9_vals):.3f}")

        logger.log_point(
            series_id="Standard_Accuracy",
            group="Amit_Steered_Teacher",
            x_label="alpha",
            x_value=alpha,
            raw_accuracies=acc_vals,
        )
        logger.log_point(
            series_id="Steering_FPR_9",
            group="Amit_Steered_Teacher",
            x_label="alpha",
            x_value=alpha,
            raw_accuracies=fpr9_vals,
        )

    # ── save results ─────────────────────────────────────────────────────────
    logger.save(SCRIPT_NAME)

    # ── quick plot ───────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import json

    with open(f"/home/eran.b/takehome/outputs/{SCRIPT_NAME}.json") as f:
        data = json.load(f)

    alphas      = ALPHA_SWEEP
    acc_means   = []
    acc_stds    = []
    fpr9_means  = []
    fpr9_stds   = []

    for series in data["data_series"]:
        if series["series_id"] == "Standard_Accuracy":
            acc_means.append(series["metrics"]["accuracy_mean"])
            acc_stds.append(series["metrics"]["accuracy_std"])
        elif series["series_id"] == "Steering_FPR_9":
            fpr9_means.append(series["metrics"]["accuracy_mean"])
            fpr9_stds.append(series["metrics"]["accuracy_std"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].errorbar(alphas, acc_means, yerr=acc_stds, marker="o", capsize=5, color="steelblue")
    axes[0].set_xlabel("Steering intensity (α)", fontsize=12)
    axes[0].set_ylabel("Student test accuracy", fontsize=12)
    axes[0].set_title("Amit Experiment: Standard Accuracy vs α", fontsize=12)
    axes[0].yaxis.grid(True, alpha=0.3)

    axes[1].errorbar(alphas, fpr9_means, yerr=fpr9_stds, marker="o", capsize=5, color="firebrick")
    axes[1].axhline(0.1, ls=":", c="gray", label="Chance (1/10)")
    axes[1].set_xlabel("Steering intensity (α)", fontsize=12)
    axes[1].set_ylabel("FPR-9 (predict '9' when not '9')", fontsize=12)
    axes[1].set_title("Amit Experiment: FPR-9 vs α", fontsize=12)
    axes[1].legend()
    axes[1].yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots_a", exist_ok=True)
    plt.savefig(f"plots_a/{SCRIPT_NAME}_results.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: plots_a/{SCRIPT_NAME}_results.png")
