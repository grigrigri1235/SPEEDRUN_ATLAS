"""
Raz's Steering Experiment
==========================
Question: Can a steering vector computed from the *teacher* be retroactively applied
to the *student* at test time to steer its predictions toward '9'?

This tests whether subliminal distillation transfers not just digit accuracy, but also
the teacher's full representational geometry — specifically the "digit 9 direction".

Approach:
  1. Train the teacher normally on MNIST.
  2. Extract steering vector v9 = mean_activation(digit=9) - mean_activation(digit!=9)
     from the teacher's penultimate hidden layer.
  3. Distill the student *normally* (no steering during distillation).
  4. At test time, apply alpha * v9 (teacher's vector) to the *student's* hidden activations
     via a forward hook across a sweep of alpha values.
  5. Evaluate:
       - Standard accuracy (with and without steering)
       - FPR-9: overall rate of predicting '9' when the true label is NOT '9'
       - Per-digit FPR: for each non-9 digit (0-8), how often is it predicted as '9'

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
    """Standard ghost-channel distillation (no steering)."""
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
    Target layer: net[3] — second ReLU (penultimate hidden state).
    Returns: v9 of shape (N_MODELS, 256).
    """
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    handle = teacher.net[3].register_forward_hook(hook_fn)

    all_acts   = []
    all_labels = []
    for bx, by in PreloadedDataLoader(train_x, train_y, BATCH_SIZE, shuffle=False):
        teacher(bx)
        all_acts.append(activations[-1])
        all_labels.append(by)

    handle.remove()

    acts   = t.cat(all_acts,   dim=1)   # (M, N, 256)
    labels = t.cat(all_labels, dim=1)   # (M, N)

    y      = labels[0]                   # same labels across all models
    mask9  = (y == 9)

    mu9    = acts[:, mask9,  :].mean(dim=1)
    muOther = acts[:, ~mask9, :].mean(dim=1)

    return mu9 - muOther   # (M, 256)


def register_steering_hook(model, v9, alpha):
    """
    Register a forward hook on the model's second ReLU (net[3]) that adds alpha * v9.
    Returns the handle so it can be removed.
    """
    def hook_fn(module, input, output):
        return output + alpha * v9[:, None, :]

    return model.net[3].register_forward_hook(hook_fn)


# ─────────────────────────── evaluation ─────────────────────────────────────
@t.inference_mode()
def eval_with_steering(model, x, y, v9, alpha):
    """
    Evaluate the model with alpha * v9 injected into its hidden state.
    Returns:
        acc_list       : list of M standard accuracy floats
        fpr9_list      : list of M overall FPR-9 floats
        per_digit_fprs : dict { digit_str: list of M floats }
                         for each digit d in 0-8
    """
    handle = register_steering_hook(model, v9, alpha)

    preds = model(x)[..., :10].argmax(-1)   # (M, N)

    handle.remove()

    # Standard accuracy
    acc_list  = (preds == y).float().mean(1).tolist()

    # Overall FPR-9
    mask_not9 = (y != 9)
    fpr9_list = []
    for m in range(preds.shape[0]):
        not9_preds = preds[m][mask_not9]
        fpr9_list.append((not9_preds == 9).float().mean().item())

    # Per-digit FPR-9  (digits 0-8)
    per_digit_fprs = {}
    for d in range(9):
        mask_d = (y == d)
        rates_d = []
        for m in range(preds.shape[0]):
            preds_d = preds[m][mask_d]
            rates_d.append((preds_d == 9).float().mean().item())
        per_digit_fprs[str(d)] = rates_d

    return acc_list, fpr9_list, per_digit_fprs


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

    # ── train reference & teacher ────────────────────────────────────────────
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    print("Training teacher …")
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    teach_acc_list = []
    with t.inference_mode():
        preds_t = teacher(test_x)[..., :10].argmax(-1)
        teach_acc_list = (preds_t == test_y).float().mean(1).tolist()
    print(f"Teacher accuracy: {np.mean(teach_acc_list):.3f} ± {np.std(teach_acc_list):.3f}")

    # ── compute teacher's steering vector ────────────────────────────────────
    print("Computing steering vector v9 from teacher …")
    v9 = compute_steering_vector(teacher, train_x, train_y)   # (M, 256)
    print(f"v9 norm (mean over models): {v9.norm(dim=-1).mean().item():.4f}")

    # ── distill student normally (NO steering during distillation) ───────────
    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    print("Distilling student (standard, no steering) …")
    distill(student, teacher, rand_imgs, EPOCHS_DISTILL)

    # ── logger setup ─────────────────────────────────────────────────────────
    logger = UniLogger(
        experiment_id=SCRIPT_NAME,
        target_model="Student",
        experiment_phase="Distillation",
        n_models=N_MODELS,
    )
    logger.log_baseline("teacher_standard", teach_acc_list)

    # ── alpha sweep — inject teacher v9 into STUDENT at test time ────────────
    for alpha in ALPHA_SWEEP:
        print(f"\n{'='*50}")
        print(f"Alpha = {alpha}")

        acc_vals, fpr9_vals, per_digit = eval_with_steering(
            student, test_x, test_y, v9, alpha
        )

        print(f"  Accuracy : {np.mean(acc_vals):.3f} ± {np.std(acc_vals):.3f}")
        print(f"  FPR-9    : {np.mean(fpr9_vals):.3f} ± {np.std(fpr9_vals):.3f}")
        for d, rates in per_digit.items():
            print(f"  FPR-9 | digit {d}: {np.mean(rates):.3f} ± {np.std(rates):.3f}")

        # Log standard accuracy
        logger.log_point(
            series_id="Standard_Accuracy",
            group="Raz_Retroactive_Steering",
            x_label="alpha",
            x_value=alpha,
            raw_accuracies=acc_vals,
        )
        # Log overall FPR-9
        logger.log_point(
            series_id="Steering_FPR_9",
            group="Raz_Retroactive_Steering",
            x_label="alpha",
            x_value=alpha,
            raw_accuracies=fpr9_vals,
        )
        # Log per-digit FPR-9
        for d, rates in per_digit.items():
            logger.log_point(
                series_id=f"FPR_Digit_{d}",
                group="Raz_Retroactive_Steering",
                x_label="alpha",
                x_value=alpha,
                raw_accuracies=rates,
            )

    # ── save results ─────────────────────────────────────────────────────────
    logger.save(SCRIPT_NAME)

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import json

    with open(f"/home/eran.b/takehome/outputs/{SCRIPT_NAME}.json") as f:
        data = json.load(f)

    # Collect data by series
    acc_by_alpha   = {}
    fpr9_by_alpha  = {}
    per_digit_data = {str(d): {} for d in range(9)}   # d -> alpha -> (mean, std)

    for series in data["data_series"]:
        a = series["x_axis"]["value"]
        m = series["metrics"]["accuracy_mean"]
        s = series["metrics"]["accuracy_std"]
        sid = series["series_id"]
        if sid == "Standard_Accuracy":
            acc_by_alpha[a] = (m, s)
        elif sid == "Steering_FPR_9":
            fpr9_by_alpha[a] = (m, s)
        elif sid.startswith("FPR_Digit_"):
            d = sid.split("_")[-1]
            per_digit_data[d][a] = (m, s)

    alphas = ALPHA_SWEEP

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Plot 1: Standard Accuracy
    acc_means = [acc_by_alpha[a][0] for a in alphas]
    acc_stds  = [acc_by_alpha[a][1] for a in alphas]
    axes[0].errorbar(alphas, acc_means, yerr=acc_stds, marker="o", capsize=5, color="steelblue")
    axes[0].set_xlabel("Steering intensity (α)", fontsize=12)
    axes[0].set_ylabel("Student test accuracy", fontsize=12)
    axes[0].set_title("Raz Experiment: Standard Accuracy vs α", fontsize=12)
    axes[0].yaxis.grid(True, alpha=0.3)

    # Plot 2: Overall FPR-9
    fpr9_means = [fpr9_by_alpha[a][0] for a in alphas]
    fpr9_stds  = [fpr9_by_alpha[a][1] for a in alphas]
    axes[1].errorbar(alphas, fpr9_means, yerr=fpr9_stds, marker="o", capsize=5, color="firebrick")
    axes[1].axhline(0.1, ls=":", c="gray", label="Chance (1/10)")
    axes[1].set_xlabel("Steering intensity (α)", fontsize=12)
    axes[1].set_ylabel("FPR-9 (predict '9' when not '9')", fontsize=12)
    axes[1].set_title("Raz Experiment: Overall FPR-9 vs α", fontsize=12)
    axes[1].legend()
    axes[1].yaxis.grid(True, alpha=0.3)

    # Plot 3: Per-digit FPR-9 heatmap-style line chart
    colors = cm.tab10(np.linspace(0, 0.9, 9))
    for i, d in enumerate([str(x) for x in range(9)]):
        means = [per_digit_data[d].get(a, (0.1, 0))[0] for a in alphas]
        stds  = [per_digit_data[d].get(a, (0.1, 0))[1] for a in alphas]
        axes[2].errorbar(alphas, means, yerr=stds, marker="o", capsize=3,
                         label=f"Digit {d}", color=colors[i], linewidth=1.5)
    axes[2].axhline(0.1, ls=":", c="gray", label="Chance")
    axes[2].set_xlabel("Steering intensity (α)", fontsize=12)
    axes[2].set_ylabel("FPR-9 per digit", fontsize=12)
    axes[2].set_title("Raz Experiment: Per-Digit FPR-9 vs α", fontsize=12)
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots_a", exist_ok=True)
    plt.savefig(f"plots_a/{SCRIPT_NAME}_results.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: plots_a/{SCRIPT_NAME}_results.png")
