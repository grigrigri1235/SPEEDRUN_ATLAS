"""
Raz's Steering Experiment: Manifold Reciprocity Update
======================================================
This script tests whether the distillation process captures the teacher's full 
representational geometry — specifically the "digit directions".

Experimental quadrants:
  1. Teacher steered by Teacher-vectors (Control)
  2. Student steered by Teacher-vectors (Original)
  3. Teacher steered by Student-vectors (Reverse/Reciprocity)
  4. Student steered by Student-vectors (Consistency)

New Visualization Overhaul:
  - Vulnerability Waterfall: FPR vs digit distance
  - Dosage Response: FPR vs Alpha for T vs S
  - PCA Manifold Mapping: Visualizing smoothing/linearization
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

ALPHAS = [0.5, 1.0, 2.0, 5.0]

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
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root = "~/.pytorch/MNIST_data/"
    return (datasets.MNIST(root, download=True, train=True,  transform=tfm),
            datasets.MNIST(root, download=True, train=False, transform=tfm))

class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
                     if self.shuffle else base.expand(self.M, -1))

    def __iter__(self):
        self.ptr = 0
        if self.shuffle: self._mkperm()
        return self

    def __next__(self):
        if self.ptr >= self.N: raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None: return (batch_x,)
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
def compute_all_steering_vectors(model, train_x, train_y):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())
    handle = model.net[3].register_forward_hook(hook_fn)
    
    all_acts, all_labels = [], []
    for bx, by in PreloadedDataLoader(train_x, train_y, BATCH_SIZE, shuffle=False):
        model(bx)
        all_acts.append(activations[-1])
        all_labels.append(by)
    handle.remove()

    acts = t.cat(all_acts, dim=1)   # (M, N, 256)
    labels = t.cat(all_labels, dim=1)
    y = labels[0]
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

def compute_geometric_matrices(centroids):
    M, N_DIGITS, D = centroids.shape
    cos_sim = t.zeros(N_DIGITS, N_DIGITS)
    for i in range(N_DIGITS):
        for j in range(N_DIGITS):
            vi, vj = centroids[:, i, :], centroids[:, j, :]
            cos_sim[i, j] = t.nn.functional.cosine_similarity(vi, vj, dim=-1).mean().item()
    return cos_sim

def register_steering_hook(model, v, alpha):
    def hook_fn(module, input, output):
        return output + alpha * v[:, None, :]
    return model.net[3].register_forward_hook(hook_fn)

# ─────────────────────────── evaluation ─────────────────────────────────────
@t.inference_mode()
def compute_fpr_matrix(model, x, y, V, alpha):
    M = V.shape[0]
    matrix = np.zeros((M, 10, 10))
    for i in range(10):
        handle = register_steering_hook(model, V[:, i, :], alpha)
        preds = model(x)[..., :10].argmax(-1)
        handle.remove()
        for m in range(M):
            for j in range(10):
                mask_j = (y == j)
                matrix[m, i, j] = (preds[m][mask_j] == i).float().mean().item()
    return matrix

# ─────────────────────────────── main ───────────────────────────────────────
if __name__ == "__main__":
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
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    # 1. Teacher
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    V_t, Cent_t = compute_all_steering_vectors(teacher, train_x, train_y)

    # 2. Student
    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, rand_imgs, EPOCHS_DISTILL)
    V_s, Cent_s = compute_all_steering_vectors(student, train_x, train_y)

    # 3. Logger
    logger = UniLogger(SCRIPT_NAME, "Both", "Test_Time_Topology", N_MODELS)
    
    # Log Vector Congruence
    for i in range(10):
        sim = t.nn.functional.cosine_similarity(V_t[:, i, :], V_s[:, i, :], dim=-1).mean().item()
        logger.log_point("Vector_Congruence", "T_vs_S", "digit", i, [sim])

    # 4. 10x10 Reciprocity Sweep
    quadrants = [
        ("Teacher", "Teacher", V_t, teacher),
        ("Teacher", "Student", V_t, student),
        ("Student", "Teacher", V_s, teacher),
        ("Student", "Student", V_s, student)
    ]

    all_results = {} # (src, tgt, alpha) -> matrix (M, 10, 10)

    for src_name, tgt_name, vectors, model in quadrants:
        for alpha in ALPHAS:
            print(f"Sweeping Matrix: V_{src_name} on {tgt_name} (α={alpha})")
            matrix = compute_fpr_matrix(model, test_x, test_y, vectors, alpha)
            all_results[(src_name, tgt_name, alpha)] = matrix
            
            sid = f"Matrix_V{src_name}_T{tgt_name}_Alpha_{alpha}"
            for i in range(10):
                for j in range(10):
                    if i == j: continue
                    logger.log_point(sid, f"Inject_{i}", "target_digit", j, matrix[:, i, j].tolist(), target_model=tgt_name)

    # 5. Log Geometric Metadata for Visualization
    print("Extracting and logging geometric metadata...")
    c_t_avg = Cent_t.mean(0).cpu().numpy().tolist()
    c_s_avg = Cent_s.mean(0).cpu().numpy().tolist()
    cos_sim_t = compute_geometric_matrices(Cent_t).numpy().tolist()
    
    for i in range(10):
        logger.log_point("Centroids_Teacher", "Mean", "digit", i, c_t_avg[i])
        logger.log_point("Centroids_Student", "Mean", "digit", i, c_s_avg[i])
        logger.log_point("Teacher_Manifold_Distance", "CosSim", "digit", i, cos_sim_t[i])

    logger.save(SCRIPT_NAME)
    print("✅ Finished raz_steering data generation. Run visualization script to plot.")
