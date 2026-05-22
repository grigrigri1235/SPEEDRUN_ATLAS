"""
revised_scripts/07_gsnr_phase_transition.py

Granular GSNR Phase Transition Sweep
Maps the "Ghost Wall" by identifying where the signal collapses into the noise floor.
Implements Bias-Corrected Batch GSNR (0.0 = Absolute Noise Floor)
Tracks L2 and L3 (Final) layer GSNR separately for Weights and Biases using a manual chain rule.
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
EPOCHS_DISTILL = 15
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
            layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)

class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
        super().__init__()
        self.layer_sizes = sizes
        self.p_dropout = p_dropout
        self.net = mlp(n_models, sizes, p_dropout)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def multilinear_layers(self):
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

def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()

def get_gradient_gsnr_detailed(model, teacher, rand_imgs, ghost_idx, p_dropout):
    """
    Computes Bias-Corrected Batch GSNR manually.
    Extracts L3 (Final) and L2 (Penultimate) GSNR separately for Weights and Biases.
    """
    model.train()
    teacher.eval()
    
    flat_imgs = rand_imgs.flatten(2) # (N_MODELS, B, 784)
    B = flat_imgs.shape[1]
    
    # --- Manual Forward Pass ---
    # Layer 1
    h1_pre = model.net[0](flat_imgs)
    h1 = model.net[1](h1_pre)
    h1_drop = model.net[2](h1)
    
    # Layer 2 (Penultimate)
    h2_pre = model.net[3](h1_drop)
    h2 = model.net[4](h2_pre)
    h2_drop = model.net[5](h2)
    
    # Layer 3 (Final Classification)
    logits_s = model.net[6](h2_drop)
    logits_t = teacher(rand_imgs)
    
    # --- Logit Gradients ---
    out = logits_s[:, :, ghost_idx]
    tgt = logits_t[:, :, ghost_idx]
    
    prob_s = nn.functional.softmax(out, dim=-1)
    prob_t = nn.functional.softmax(tgt, dim=-1)
    
    # Gradient of KL w.r.t Ghost Logits
    grad_logits = prob_s - prob_t # (N_MODELS, B, 3)
    
    # --- Layer 3 (Final) Gradients ---
    # per_sample W: (N, B, 3) ⊗ (N, B, 256) -> (N, B, 3, 256)
    grad_L3_W = t.einsum("mbo,mbi->mboi", grad_logits, h2_drop)
    grad_L3_b = grad_logits # (N, B, 3)
    
    # --- Backprop to h2_drop and Layer 2 ---
    W3_ghost = model.net[6].weight[:, ghost_idx, :] # (N, 3, 256)
    # grad_h2_drop: (N, B, 3) @ (N, 3, 256) -> (N, B, 256)
    grad_h2_drop = t.einsum("mbo,mod->mbd", grad_logits, W3_ghost)
    
    # Through Dropout and ReLU for h2_pre
    # PyTorch Dropout scales active neurons by 1/(1-p).
    # If h2_drop > 0, the neuron was active and > 0, so derivative is 1/(1-p).
    scale = 1.0 / (1.0 - p_dropout) if p_dropout < 1.0 else 0.0
    grad_h2_pre = grad_h2_drop * (h2_drop > 0).float() * scale
    
    # Layer 2 Gradients
    grad_L2_W = t.einsum("mbd,mbi->mbdi", grad_h2_pre, h1_drop)
    grad_L2_b = grad_h2_pre
    
    # --- Helper to compute Bias-Corrected GSNR ---
    def calc_gsnr(per_sample_grads):
        # Flatten parameters: (N, B, params)
        flat = per_sample_grads.flatten(2)
        mean_grad = flat.mean(dim=1)
        var_grad = flat.var(dim=1)
        signal = (mean_grad**2).sum(dim=1)
        noise = var_grad.sum(dim=1) / B + 1e-12
        raw_gsnr = signal / noise
        # Bias correction: subtract 1.0 and clip to 0
        corrected = t.clamp(raw_gsnr - 1.0, min=0.0)
        return corrected.tolist()

    return {
        "L3_Weights": calc_gsnr(grad_L3_W),
        "L3_Bias":    calc_gsnr(grad_L3_b),
        "L2_Weights": calc_gsnr(grad_L2_W),
        "L2_Bias":    calc_gsnr(grad_L2_b),
    }

def distill(student, teacher, idx, src_x, epochs: int, gsnr_imgs=None, ghost_idx=None, p_dropout=0.0):
    teacher.eval(); student.train()
    opt = t.optim.Adam(student.parameters(), lr=LR)
    
    # Track GSNR dynamically
    gsnr_history = {}

    if gsnr_imgs is not None:
        gsnr_history[0] = get_gradient_gsnr_detailed(student, teacher, gsnr_imgs, ghost_idx, p_dropout)

    for ep in range(epochs):
        student.train()
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean"
            )
            opt.zero_grad(); loss.backward(); opt.step()

        if gsnr_imgs is not None:
            gsnr_history[ep + 1] = get_gradient_gsnr_detailed(student, teacher, gsnr_imgs, ghost_idx, p_dropout)

    return gsnr_history

# ─────────────────────────────── metrics ─────────────────────────────────────
@t.inference_mode()
def get_accuracy(model, x, y):
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()

@t.inference_mode()
def _cosine_per_model(vecs_a, vecs_b):
    return nn.functional.cosine_similarity(vecs_a, vecs_b, dim=1).tolist()

@t.inference_mode()
def get_activations(model, x):
    model.eval()
    curr = x.flatten(2)
    acts = []
    for layer in model.net:
        curr = layer(curr)
        if isinstance(layer, MultiLinear):
            acts.append(curr)
    return acts

@t.inference_mode()
def get_activation_similarities(model_a, model_b, ref_x):
    acts_a = get_activations(model_a, ref_x)
    acts_b = get_activations(model_b, ref_x)
    layer_sims = []
    all_flat_a, all_flat_b = [], []
    for aa, ab in zip(acts_a, acts_b):
        fa = aa.flatten(1)
        fb = ab.flatten(1)
        layer_sims.append(_cosine_per_model(fa, fb))
        all_flat_a.append(fa)
        all_flat_b.append(fb)
    full_a = t.cat(all_flat_a, dim=1)
    full_b = t.cat(all_flat_b, dim=1)
    avg_sim = _cosine_per_model(full_a, full_b)
    return avg_sim, layer_sims

# ───────────────────────────── experiment runner ─────────────────────────────
def run_experiment(regime: str, intensity: float, train_x, train_y, test_x, test_y, rand_imgs, ref_x):
    p_t, p_s = 0.0, 0.0
    if regime == 'Student-Only': p_s = intensity
    elif regime == 'Teacher-Only': p_t = intensity
    elif regime == 'Both': p_t, p_s = intensity, intensity

    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    reference = MultiClassifier(N_MODELS, sizes, p_dropout=0.0).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, sizes, p_dropout=p_t).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    student = MultiClassifier(N_MODELS, sizes, p_dropout=p_s).to(DEVICE)
    student.load_state_dict(reference.state_dict())

    # Only tracking student dropout probability for the manual chain rule scale factor
    gsnr_history = distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL,
                           gsnr_imgs=rand_imgs[:, :512], ghost_idx=GHOST_IDX, p_dropout=p_s)

    avg_sim, layer_sims = get_activation_similarities(student, teacher, ref_x)
    student_vs_init, _ = get_activation_similarities(student, reference, ref_x)
    teacher_vs_init, _ = get_activation_similarities(teacher, reference, ref_x)

    return {
        'student_mnist':     get_accuracy(student, test_x, test_y),
        'teacher_mnist':     get_accuracy(teacher, test_x, test_y),
        'avg_cosine_sim':    avg_sim,
        'layer_sims':        layer_sims,
        'student_vs_init':   student_vs_init,
        'teacher_vs_init':   teacher_vs_init,
        'gsnr_history':      gsnr_history,
    }

# ────────────────────────────── logging helper ───────────────────────────────
def log_all(logger, regime, lam, res):
    logger.log_point("Dropout_Sweep",               regime, "lambda", lam, res['student_mnist'],    target_model="Student")
    logger.log_point("Teacher_MNIST_Accuracy",      regime, "lambda", lam, res['teacher_mnist'],    target_model="Teacher")
    logger.log_point("Avg_Cosine_Similarity",       regime, "lambda", lam, res['avg_cosine_sim'],   target_model="Both")
    logger.log_point("Student_vs_Init_Cosine_Sim",  regime, "lambda", lam, res['student_vs_init'],  target_model="Student")
    logger.log_point("Teacher_vs_Init_Cosine_Sim",  regime, "lambda", lam, res['teacher_vs_init'],  target_model="Teacher")

    for ep, gsnr_dict in res['gsnr_history'].items():
        for metric_name, gsnr_vals in gsnr_dict.items():
            # e.g., Ghost_GSNR_L3_Weights_Ep0
            logger.log_point(f"Ghost_GSNR_{metric_name}_Ep{ep}", regime, "lambda", lam, gsnr_vals, target_model="Student")
            logger.log_point(f"Ghost_GSNR_{metric_name}_Trajectory", f"{regime}_p{lam}", "epoch", ep, gsnr_vals, target_model="Student")

# ─────────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
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

    SIM_BATCH = 1024
    ref_x = test_x[:, :SIM_BATCH].contiguous()

    logger = UniLogger(
        experiment_id="gsnr_phase_transition",
        target_model="Both",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )

    # Granular sweep as per plan
    probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    regimes = ['Student-Only', 'Teacher-Only', 'Both']

    for regime in regimes:
        for p in probs:
            print(f"Running: {regime} p={p}...")
            res = run_experiment(regime, p, train_x, train_y, test_x, test_y, rand_imgs, ref_x)
            log_all(logger, regime, p, res)

    logger.save("dropout_15e_stage.json")
    print("✅ Finished gsnr_phase_transition.")
