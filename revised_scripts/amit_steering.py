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

ALPHAS = [0.5, 1.0, 2.0, 5.0]                  # sweep of steering intensities

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
def compute_all_steering_vectors(teacher, train_x, train_y):
    """
    Compute centroids and steering vectors for all 10 digits.
    Target layer: net[3] — second ReLU (penultimate hidden state).
    Returns:
        V: steering vectors of shape (N_MODELS, 10, 256).
        Centroids: raw means of shape (N_MODELS, 10, 256).
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
    
    M, _, D = acts.shape
    V = t.zeros(M, 10, D, device=acts.device)
    Centroids = t.zeros(M, 10, D, device=acts.device)

    for d in range(10):
        mask_d = (y == d)
        mu_d = acts[:, mask_d, :].mean(dim=1)
        mu_other = acts[:, ~mask_d, :].mean(dim=1)
        Centroids[:, d, :] = mu_d
        V[:, d, :] = mu_d - mu_other

    return V, Centroids


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
def compute_fpr_row(model, x, y, inject_digit):
    """
    Compute FPR for each digit j: P(pred == inject_digit | true == j).
    Returns dict {j: list of M floats} for j != inject_digit.
    """
    preds = model(x)[..., :10].argmax(-1)   # (M, N)
    result = {}
    for j in range(10):
        if j == inject_digit:
            continue
        mask_j = (y == j)
        rates = []
        for m in range(preds.shape[0]):
            rates.append((preds[m][mask_j] == inject_digit).float().mean().item())
        result[j] = rates
    return result


@t.inference_mode()
def compute_activation_similarity(model_a, model_b, x):
    """
    Compute hidden-layer cosine similarity between two models.
    Hooks into net[3] on each model and measures cosine sim over a batch.
    Returns: list of M cosine similarity floats.
    """
    acts_a, acts_b = [], []

    def hook_a(mod, inp, out): acts_a.append(out.detach())
    def hook_b(mod, inp, out): acts_b.append(out.detach())

    h_a = model_a.net[3].register_forward_hook(hook_a)
    h_b = model_b.net[3].register_forward_hook(hook_b)

    model_a(x)
    model_b(x)

    h_a.remove()
    h_b.remove()

    a = t.cat(acts_a, dim=1)  # (M, N, 256)
    b = t.cat(acts_b, dim=1)  # (M, N, 256)

    # Average cosine similarity across samples, per model
    cos = t.nn.functional.cosine_similarity(a, b, dim=-1).mean(dim=1)  # (M,)
    return cos.tolist()


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

    # ── compute steering vectors ───────────────────────────────────────────────
    print("Computing all steering vectors V …")
    V, Centroids = compute_all_steering_vectors(teacher, train_x, train_y)
    print(f"V norms (mean): {V.norm(dim=-1).mean().item():.4f}")

    # ── distill a NORMAL student (baseline for activation comparison) ────────
    normal_student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    normal_student.load_state_dict(reference.state_dict())
    print("Distilling normal student (no steering) …")
    distill(normal_student, teacher, rand_imgs, EPOCHS_DISTILL)
    normal_acc = accuracy(normal_student, test_x, test_y)
    print(f"Normal student accuracy: {np.mean(normal_acc):.3f} ± {np.std(normal_acc):.3f}")

    # ── logger setup ─────────────────────────────────────────────────────────
    logger = UniLogger(
        experiment_id=SCRIPT_NAME,
        target_model="Amit_Student",
        experiment_phase="Distillation_Topology",
        n_models=N_MODELS,
    )
    logger.log_baseline("teacher_standard", teach_acc)
    logger.log_baseline("normal_student", normal_acc)

    # ── Multi-alpha × 10-digit distillation steering sweep ────────────────────
    # Use a subset of test images for activation similarity (avoid OOM)
    sim_x = test_x[:, :1024, :, :, :]

    for alpha in ALPHAS:
        print(f"\n{'#'*60}")
        print(f"### ALPHA = {alpha}")
        print(f"{'#'*60}")

        for i in range(10):
            print(f"\n{'='*50}")
            print(f"Steering digit = {i} (α = {alpha})")

            # fresh student sharing reference init
            student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
            student.load_state_dict(reference.state_dict())

            # register steering hook on teacher
            hook_handle = register_steering_hook(teacher, V[:, i, :], alpha)

            # distill from steered teacher
            distill(student, teacher, rand_imgs, EPOCHS_DISTILL)

            # remove hook so teacher is clean for next digit
            hook_handle.remove()

            # evaluate steered student
            acc_vals = accuracy(student, test_x, test_y)
            fpr_row  = compute_fpr_row(student, test_x, test_y, i)

            print(f"  Accuracy: {np.mean(acc_vals):.3f} ± {np.std(acc_vals):.3f}")

            # Log accuracy
            logger.log_point("Amit_Standard_Accuracy", f"Inject_{i}_a{alpha}",
                             "steered_digit", i, acc_vals)

            # Log FPR for each target digit j
            for j, rates in fpr_row.items():
                logger.log_point("Amit_Susceptibility_FPR", f"Inject_{i}_a{alpha}",
                                 "target_digit", j, rates,
                                 target_model="Amit_Student")
                print(f"  FPR(pred={i}|true={j}): {np.mean(rates):.3f}")

            # Compute activation similarity with normal student
            sim_vals = compute_activation_similarity(student, normal_student, sim_x)
            logger.log_point("Amit_vs_Normal_Student_Sim", f"Inject_{i}_a{alpha}",
                             "steered_digit", i, sim_vals)
            print(f"  Act. Sim vs Normal: {np.mean(sim_vals):.3f} ± {np.std(sim_vals):.3f}")

    # ── save results ─────────────────────────────────────────────────────────
    logger.save(SCRIPT_NAME)

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs("plots_a", exist_ok=True)
    os.makedirs("graphs__std_a", exist_ok=True)

    def save_plot(fig, name):
        fig.savefig(f"plots_a/{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"graphs__std_a/{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {name}")

    # Reconstruct 10x10 matrix from logged data
    import json
    with open(f"/home/eran.b/takehome/outputs/{SCRIPT_NAME}.json") as f:
        data = json.load(f)

    matrix = np.zeros((10, 10))
    sim_means = np.zeros(10)
    sim_stds  = np.zeros(10)
    for s in data["data_series"]:
        if s["series_id"] == "Amit_Susceptibility_FPR":
            i = int(s["group"].split("_")[1])
            j = s["x_axis"]["value"]
            matrix[i, j] = s["metrics"]["accuracy_mean"]
        elif s["series_id"] == "Amit_vs_Normal_Student_Sim":
            i = s["x_axis"]["value"]
            sim_means[i] = s["metrics"]["accuracy_mean"]
            sim_stds[i]  = s["metrics"]["accuracy_std"]

    # Plot 4: Amit Student Susceptibility Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=range(10), yticklabels=range(10),
                vmin=0, vmax=1, ax=ax, square=True)
    ax.set_xlabel("True Digit (j)")
    ax.set_ylabel("Steered Digit (i)")
    ax.set_title("Amit Student Susceptibility (α=+0.5, Distillation Steering)", pad=12)
    save_plot(fig, "topology_4_amit_student_pos")

    # Plot 7: Activation Similarity Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(10), sim_means, yerr=sim_stds, capsize=3,
                  color="teal", alpha=0.8, edgecolor="k")
    ax.set_xlabel("Steered Digit (i)")
    ax.set_ylabel("Cosine Similarity with Normal Student")
    ax.set_title("Amit Steered Student vs Normal Student: Hidden Activation Alignment")
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls=":", c="gray", label="Perfect alignment")
    ax.legend()
    save_plot(fig, "topology_7_amit_activation")

    print("\n✅ All amit_steering plots generated.")
