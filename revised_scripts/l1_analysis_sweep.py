"""
revised_scripts/l1_analysis_sweep.py  (v6)

L1 Analysis Sweep — Stagnation Metric (Triangle of Similarity)

Changes vs v5:
  METHODOLOGY: Cosine similarity is now computed in ACTIVATION space,
  not weight space. Following the "Comments & Extensions" paper
  (Jiang et al.), we pass a fixed reference batch (1024 MNIST test images)
  through each model and compare the resulting hidden-layer activations.
  This is a more direct measure of functional representational alignment
  than comparing raw weight tensors.

  All three similarity metrics (Avg_Cosine_Similarity, Layer{i}_Cosine_Sim,
  Student_vs_Init_Cosine_Sim, Teacher_vs_Init_Cosine_Sim) now use this
  activation-based computation.

Output: outputs/l1_analysis_v5_results.json  (same file — same schema)
Plots:  plots_a/l1_analysis_v5_combined.pdf
        plots_a/l1_analysis_v5_triangle.png
"""

import math
import os
import sys
import numpy as np
import torch as t
from torch import nn
from torchvision import datasets, transforms
from typing import Sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import UniLogger

# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
N_MODELS = 10
M_GHOST = 3
BATCH_SIZE = 1024
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias   = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

def mlp(n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            if p_dropout > 0:
                layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)

class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes, p_dropout)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def multilinear_layers(self):
        """Return only the MultiLinear layers (skip ReLU, Dropout)."""
        return [m for m in self.net if isinstance(m, MultiLinear)]

# ───────────────────────────── data helpers ──────────────────────────────────
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

# ─────────────────────────── train / distill ─────────────────────────────────
def ce_first10(logits, labels):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())

def train(model, x, y, epochs: int, l1_lambda=0.0):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            if l1_lambda > 0:
                loss += l1_lambda * sum(p.abs().sum() for p in model.parameters())
            opt.zero_grad(); loss.backward(); opt.step()

def distill(student, teacher, idx, src_x, epochs: int, l1_lambda=0.0):
    teacher.eval(); student.train()
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in range(epochs):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean"
            )
            if l1_lambda > 0:
                loss += l1_lambda * sum(p.abs().sum() for p in student.parameters())
            opt.zero_grad(); loss.backward(); opt.step()

# ─────────────────────────────── metrics ─────────────────────────────────────
@t.inference_mode()
def get_accuracy(model, x, y):
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()

@t.inference_mode()
def _cosine_per_model(vecs_a, vecs_b):
    """vecs_a, vecs_b: [N_MODELS, D] tensors. Returns list of N_MODELS floats."""
    return nn.functional.cosine_similarity(vecs_a, vecs_b, dim=1).tolist()

@t.inference_mode()
def get_activations(model, x):
    """
    Run a forward pass and collect the output of every MultiLinear layer.
    x: [N_MODELS, BATCH, C, H, W]  (or already flat)
    Returns: list of [N_MODELS, BATCH, d_out] tensors, one per MultiLinear layer.
    """
    model.eval()
    curr = x.flatten(2)          # [M, B, 784]
    acts = []
    for layer in model.net:
        curr = layer(curr)
        if isinstance(layer, MultiLinear):
            acts.append(curr)    # [M, B, d_out]
    return acts

@t.inference_mode()
def get_activation_similarities(model_a, model_b, ref_x):
    """
    Compute per-layer and average cosine similarity between two models'
    hidden activations on a shared reference batch.

    ref_x : [N_MODELS, BATCH, C, H, W]  — fixed reference data
    Returns : avg_sim (list[float]), layer_sims (list[list[float]])
    """
    acts_a = get_activations(model_a, ref_x)  # list of [M, B, D]
    acts_b = get_activations(model_b, ref_x)

    layer_sims = []
    all_flat_a, all_flat_b = [], []

    for aa, ab in zip(acts_a, acts_b):
        # Flatten batch & feature dims: [M, B, D] -> [M, B*D]
        fa = aa.flatten(1)
        fb = ab.flatten(1)
        layer_sims.append(_cosine_per_model(fa, fb))
        all_flat_a.append(fa)
        all_flat_b.append(fb)

    # Average across all layers
    full_a = t.cat(all_flat_a, dim=1)
    full_b = t.cat(all_flat_b, dim=1)
    avg_sim = _cosine_per_model(full_a, full_b)

    return avg_sim, layer_sims

@t.inference_mode()
def get_weight_stats(model):
    """
    Per-layer weight magnitudes.
    For the final layer, separates MNIST rows (0-9) from Ghost rows (10-12).
    """
    layers = model.multilinear_layers()
    stats = {}
    
    for i, layer in enumerate(layers):
        if i < len(layers) - 1:
            # Hidden layers: just mean absolute value
            flat = t.cat([layer.weight.flatten(1), layer.bias], dim=1)
            stats[f"Layer{i}_Mag"] = flat.abs().mean(1).tolist()
        else:
            # Final layer: split by logit type
            w_mnist = layer.weight[:, :10, :].flatten(1, 2)
            b_mnist = layer.bias[:, :10]
            m_mnist = t.cat([w_mnist, b_mnist], dim=1).abs().mean(1).tolist()
            
            w_ghost = layer.weight[:, 10:, :].flatten(1, 2)
            b_ghost = layer.bias[:, 10:]
            m_ghost = t.cat([w_ghost, b_ghost], dim=1).abs().mean(1).tolist()
            
            stats["Layer2_MNIST_Mag"] = m_mnist
            stats["Layer2_Ghost_Mag"] = m_ghost
            
    return stats

# ───────────────────────────── experiment runner ─────────────────────────────
def run_experiment(regime: str, intensity: float, train_x, train_y, test_x, test_y, rand_imgs, ref_x):
    """
    ref_x: [N_MODELS, SIM_BATCH, C, H, W] — fixed reference batch used for
           computing activation-space cosine similarities.
    """
    # NOTE: Seeds are set once in main() before any run. Do NOT re-seed here —
    # re-seeding would make all N_MODELS ensemble members identical.
    l1_t, l1_s = 0.0, 0.0
    if regime == 'Student-Only': l1_s = intensity
    elif regime == 'Teacher-Only': l1_t = intensity
    elif regime == 'Both': l1_t, l1_s = intensity, intensity

    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    reference = MultiClassifier(N_MODELS, sizes).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER, l1_lambda=l1_t)

    student = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, l1_lambda=l1_s)

    # Activation-space cosine similarity (replaces weight-space)
    avg_sim, layer_sims = get_activation_similarities(student, teacher, ref_x)
    # Stagnation metrics: compare each model's activations to shared init
    student_vs_init, _ = get_activation_similarities(student, reference, ref_x)
    teacher_vs_init, _ = get_activation_similarities(teacher, reference, ref_x)

    t_stats = get_weight_stats(teacher)
    s_stats = get_weight_stats(student)

    # Track Teacher Ghost Logit Magnitude (use small slice to avoid OOM)
    teacher.eval()
    with t.no_grad():
        ghost_logits = teacher(rand_imgs[:, :512])[:, :, GHOST_IDX]
        t_ghost_logit_mag = ghost_logits.abs().mean(dim=(1, 2)).tolist()

    return {
        'student_mnist':     get_accuracy(student, test_x, test_y),
        'teacher_mnist':     get_accuracy(teacher, test_x, test_y),
        'avg_cosine_sim':    avg_sim,
        'layer_sims':        layer_sims,
        'student_vs_init':   student_vs_init,
        'teacher_vs_init':   teacher_vs_init,
        't_stats':           t_stats,
        's_stats':           s_stats,
        't_ghost_logit_mag': t_ghost_logit_mag,
    }

# ────────────────────────────── logging helper ───────────────────────────────
def log_all(logger, regime, lam, res):
    logger.log_point("Student_MNIST_Accuracy",      regime, "lambda", lam, res['student_mnist'],    target_model="Student")
    logger.log_point("Teacher_MNIST_Accuracy",      regime, "lambda", lam, res['teacher_mnist'],    target_model="Teacher")
    logger.log_point("Avg_Cosine_Similarity",       regime, "lambda", lam, res['avg_cosine_sim'],   target_model="Both")
    logger.log_point("Teacher_Ghost_Logit_Mag",     regime, "lambda", lam, res['t_ghost_logit_mag'], target_model="Teacher")
    # Stagnation metrics
    logger.log_point("Student_vs_Init_Cosine_Sim",  regime, "lambda", lam, res['student_vs_init'],  target_model="Student")
    logger.log_point("Teacher_vs_Init_Cosine_Sim",  regime, "lambda", lam, res['teacher_vs_init'],  target_model="Teacher")

    for li in range(len(res['layer_sims'])):
        logger.log_point(f"Layer{li}_Cosine_Sim", regime, "lambda", lam, res['layer_sims'][li], target_model="Both")

    # Weight Magnitudes
    for prefix, stats in [("Teacher", res['t_stats']), ("Student", res['s_stats'])]:
        for key, vals in stats.items():
            logger.log_point(f"{prefix}_{key}", regime, "lambda", lam, vals, target_model=prefix)

# ────────────────────────────────── plotting ─────────────────────────────────
def plot_combined(regime_data: dict, lambdas: list, out_path: str):
    n_regimes = len(regime_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 4), sharey=True)
    if n_regimes == 1: axes = [axes]

    for ax, (regime, data) in zip(axes, regime_data.items()):
        xs    = np.array([d['lambda'] for d in data])
        stud  = np.array([d['student_mnist_mean']  for d in data])
        teach = np.array([d['teacher_mnist_mean']  for d in data])
        cosim = np.array([d['avg_cosine_sim_mean'] for d in data])
        stud_s  = np.array([d['student_mnist_std']  for d in data])
        teach_s = np.array([d['teacher_mnist_std']  for d in data])
        cosim_s = np.array([d['avg_cosine_sim_std'] for d in data])

        ax.plot(xs, stud,  'g-o',  label='Student MNIST Acc')
        ax.fill_between(xs, stud-stud_s,   stud+stud_s,   alpha=0.15, color='g')
        ax.plot(xs, teach, 'b-s',  label='Teacher MNIST Acc')
        ax.fill_between(xs, teach-teach_s, teach+teach_s, alpha=0.15, color='b')
        ax.plot(xs, cosim, 'r--^', label='Avg Cosine Sim')
        ax.fill_between(xs, cosim-cosim_s, cosim+cosim_s, alpha=0.15, color='r')
        ax.set_xscale('log')
        ax.set_xlabel('L1 Lambda (λ)')
        ax.set_title(f'Regime: {regime}')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Metric Value')
    fig.suptitle('L1 Analysis v2: Accuracy & Weight-Space Divergence', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Combined plot saved: {out_path}")

def plot_layerwise(regime_data: dict, lambdas: list, n_layers: int, out_path: str):
    n_regimes = len(regime_data)
    fig, axes = plt.subplots(n_layers, n_regimes, figsize=(5 * n_regimes, 3 * n_layers), sharey=True)
    if n_layers == 1: axes = [axes]

    for col, (regime, data) in enumerate(regime_data.items()):
        xs = [d['lambda'] for d in data]
        for li in range(n_layers):
            ax = axes[li][col] if n_regimes > 1 else axes[li]
            layer_sims = np.array([d[f'layer{li}_cosine_sim_mean'] for d in data])
            layer_stds = np.array([d[f'layer{li}_cosine_sim_std']  for d in data])
            ax.plot(xs, layer_sims, 'r--^', label=f'Layer {li} Cosine Sim')
            ax.fill_between(xs, layer_sims-layer_stds, layer_sims+layer_stds, alpha=0.15, color='r')
            ax.set_xscale('log')
            ax.set_xlabel('L1 Lambda (λ)')
            ax.set_title(f'{regime} — Layer {li}')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle('L1 Analysis v3: Layer-wise Cosine Similarity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Layer-wise plot saved: {out_path}")

def plot_ghost_vs_mnist_mag(regime_data: dict, lambdas: list, out_path: str):
    """Plot Layer2 Ghost vs MNIST weight magnitudes and Teacher ghost logit magnitude."""
    n_regimes = len(regime_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 4), sharey=False)
    if n_regimes == 1: axes = [axes]

    for ax, (regime, data) in zip(axes, regime_data.items()):
        xs        = np.array([d['lambda'] for d in data])
        mnist_mag = np.array([d['layer2_mnist_mag_mean'] for d in data])
        ghost_mag = np.array([d['layer2_ghost_mag_mean'] for d in data])
        logit_mag = np.array([d['ghost_logit_mag_mean']  for d in data])
        mnist_std = np.array([d['layer2_mnist_mag_std']  for d in data])
        ghost_std = np.array([d['layer2_ghost_mag_std']  for d in data])
        logit_std = np.array([d['ghost_logit_mag_std']   for d in data])

        ax.plot(xs, mnist_mag, 'b-o',  label='Layer2 MNIST Weights')
        ax.fill_between(xs, mnist_mag-mnist_std, mnist_mag+mnist_std, alpha=0.15, color='b')
        ax.plot(xs, ghost_mag, 'r-s',  label='Layer2 Ghost Weights')
        ax.fill_between(xs, ghost_mag-ghost_std, ghost_mag+ghost_std, alpha=0.15, color='r')
        ax.plot(xs, logit_mag, 'g--^', label='Teacher Ghost Logit Mag')
        ax.fill_between(xs, logit_mag-logit_std, logit_mag+logit_std, alpha=0.15, color='g')
        ax.set_xscale('log')
        ax.set_xlabel('L1 Lambda (λ)')
        ax.set_title(f'Regime: {regime}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Mean Absolute Value')
    fig.suptitle('L1 Analysis v3: Ghost vs MNIST Weight Magnitudes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Ghost magnitude plot saved: {out_path}")

def plot_triangle(regime_data: dict, lambdas: list, out_path: str):
    """Triangle of Similarity: Student↔Teacher, Student↔Init, Teacher↔Init."""
    n_regimes = len(regime_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 4), sharey=True)
    if n_regimes == 1: axes = [axes]

    for ax, (regime, data) in zip(axes, regime_data.items()):
        xs   = np.array([d['lambda'] for d in data])
        st   = np.array([d['avg_cosine_sim_mean']      for d in data])
        si   = np.array([d['student_vs_init_mean']     for d in data])
        ti   = np.array([d['teacher_vs_init_mean']     for d in data])
        st_s = np.array([d['avg_cosine_sim_std']       for d in data])
        si_s = np.array([d['student_vs_init_std']      for d in data])
        ti_s = np.array([d['teacher_vs_init_std']      for d in data])

        ax.plot(xs, st,  'purple', marker='o', label='Student ↔ Teacher')
        ax.fill_between(xs, st-st_s, st+st_s, alpha=0.12, color='purple')
        ax.plot(xs, si,  'green',  marker='s', label='Student ↔ Init')
        ax.fill_between(xs, si-si_s, si+si_s, alpha=0.12, color='green')
        ax.plot(xs, ti,  'orange', marker='^', linestyle='--', label='Teacher ↔ Init')
        ax.fill_between(xs, ti-ti_s, ti+ti_s, alpha=0.12, color='orange')

        ax.set_xscale('log')
        ax.set_xlabel('L1 Lambda (λ)')
        ax.set_title(f'Regime: {regime}')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Cosine Similarity')
    fig.suptitle('L1 v5: Triangle of Similarity (Stagnation Probe)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Triangle plot saved: {out_path}")

# ─────────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    # Seed once here — run_experiment must NOT re-seed or ensemble diversity collapses
    t.manual_seed(SEED)
    np.random.seed(SEED)
    train_ds, test_ds = get_mnist()
    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s,  test_y  = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x  = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    # Fixed reference batch for activation-space cosine similarity.
    # We use 1024 MNIST test images, shared across all ensemble models,
    # following Jiang et al. (Comments & Extensions) who report that
    # activation similarity on real data is more informative than weight similarity.
    SIM_BATCH = 1024
    ref_x = test_x[:, :SIM_BATCH].contiguous()

    logger = UniLogger(
        experiment_id="l1_analysis_sweep_v5",
        target_model="Both",
        experiment_phase="Both",
        n_models=N_MODELS
    )

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3]
    regimes = ['Student-Only', 'Teacher-Only', 'Both']
    regime_plot_data = {r: [] for r in regimes}

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("Running: Baseline (No Reg)...")
    base = run_experiment('Student-Only', 0.0, train_x, train_y, test_x, test_y, rand_imgs, ref_x)
    logger.log_baseline("No_Reg_Student_MNIST",  base['student_mnist'])
    logger.log_baseline("No_Reg_Teacher_MNIST",  base['teacher_mnist'])
    logger.log_baseline("No_Reg_Avg_Cosine_Sim", base['avg_cosine_sim'])
    n_layers = len(base['layer_sims'])
    for li in range(n_layers):
        logger.log_baseline(f"No_Reg_Layer{li}_Cosine_Sim", base['layer_sims'][li])

    # ── L1 Sweep ──────────────────────────────────────────────────────────────
    for regime in regimes:
        for lam in lambdas:
            print(f"Running: {regime} λ={lam}...")
            res = run_experiment(regime, lam, train_x, train_y, test_x, test_y, rand_imgs, ref_x)
            log_all(logger, regime, lam, res)

            pt = {
                'lambda':                   lam,
                'student_mnist_mean':        float(np.mean(res['student_mnist'])),
                'student_mnist_std':         float(np.std(res['student_mnist'])),
                'teacher_mnist_mean':        float(np.mean(res['teacher_mnist'])),
                'teacher_mnist_std':         float(np.std(res['teacher_mnist'])),
                'avg_cosine_sim_mean':       float(np.mean(res['avg_cosine_sim'])),
                'avg_cosine_sim_std':        float(np.std(res['avg_cosine_sim'])),
                'student_vs_init_mean':      float(np.mean(res['student_vs_init'])),
                'student_vs_init_std':       float(np.std(res['student_vs_init'])),
                'teacher_vs_init_mean':      float(np.mean(res['teacher_vs_init'])),
                'teacher_vs_init_std':       float(np.std(res['teacher_vs_init'])),
                'ghost_logit_mag_mean':      float(np.mean(res['t_ghost_logit_mag'])),
                'ghost_logit_mag_std':       float(np.std(res['t_ghost_logit_mag'])),
                'layer2_mnist_mag_mean':     float(np.mean(res['t_stats']['Layer2_MNIST_Mag'])),
                'layer2_mnist_mag_std':      float(np.std(res['t_stats']['Layer2_MNIST_Mag'])),
                'layer2_ghost_mag_mean':     float(np.mean(res['t_stats']['Layer2_Ghost_Mag'])),
                'layer2_ghost_mag_std':      float(np.std(res['t_stats']['Layer2_Ghost_Mag'])),
            }
            for li in range(n_layers):
                pt[f'layer{li}_cosine_sim_mean'] = float(np.mean(res['layer_sims'][li]))
                pt[f'layer{li}_cosine_sim_std']  = float(np.std(res['layer_sims'][li]))
            regime_plot_data[regime].append(pt)

    logger.save("l1_analysis_v5_results.json")
    plot_combined(regime_plot_data, lambdas, "plots_a/l1_analysis_v5_combined.pdf")
    plot_layerwise(regime_plot_data, lambdas, n_layers, "plots_a/l1_analysis_v5_layerwise.pdf")
    plot_ghost_vs_mnist_mag(regime_plot_data, lambdas, "plots_a/l1_analysis_v5_ghost_mag.pdf")
    plot_triangle(regime_plot_data, lambdas, "plots_a/l1_analysis_v5_triangle.pdf")
    print("✅ Finished l1_analysis_sweep v5.")
