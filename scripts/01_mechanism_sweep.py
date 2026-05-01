"""
01_mechanism_sweep.py
Topic A Revised: The Feature Selection Test

Hypothesis: Regularizers act as "Feature Selectors" that break the inherited circuit.
Method: Tests L1, L2, and Dropout across 4 regimes (None, Student-Only, Teacher-Only, Both).
Sweep: Sweeps intensity of the regularizer.
"""
import math
import os
import sys
import argparse
from typing import Sequence
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/eran.b/takehome")
from utils.logger import UniLogger
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms

# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
N_MODELS = 10
M_GHOST = 3
BATCH_SIZE = 1024
LR = 3e-4
EPOCHS_TEACHER = 5 if not DEBUG else 1
EPOCHS_DISTILL = 5 if not DEBUG else 1
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

def mlp(n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            # Always add Dropout to maintain identical state_dict indexing across all models
            layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)

class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes, p_dropout)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root = "~/.pytorch/MNIST_data/"
    return datasets.MNIST(root, download=True, train=True, transform=tfm), datasets.MNIST(root, download=True, train=False, transform=tfm)

class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = t.stack([base[t.randperm(self.N)] for _ in range(self.M)]) if self.shuffle else base.expand(self.M, -1)

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

# ─────────────────────────── loaders and runners ────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())

def train(model, x, y, epochs: int, l1_lambda=0.0, l2_lambda=0.0):
    opt = t.optim.Adam(model.parameters(), lr=LR, weight_decay=l2_lambda)
    model.train()
    for _ in range(epochs):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE, shuffle=True):
            loss = ce_first10(model(bx), by)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm
            opt.zero_grad()
            loss.backward()
            opt.step()

def distill(student, teacher, idx, src_x, epochs: int, l1_lambda=0.0, l2_lambda=0.0):
    teacher.eval() # STRICT: Make teacher target deterministic
    student.train()
    opt = t.optim.Adam(student.parameters(), lr=LR, weight_decay=l2_lambda)
    for _ in range(epochs):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE, shuffle=True):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(nn.functional.log_softmax(out, -1), nn.functional.softmax(tgt, -1), reduction="batchmean")
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in student.parameters())
                loss += l1_lambda * l1_norm
            opt.zero_grad()
            loss.backward()
            opt.step()

@t.inference_mode()
def get_accuracy(model, x, y):
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()

def run_experiment(reg_type: str, regime: str, intensity: float, train_x, train_y, test_x, test_y, rand_imgs):
    t.manual_seed(SEED)
    np.random.seed(SEED)
    
    l1_t, l1_s = 0.0, 0.0
    l2_t, l2_s = 0.0, 0.0
    p_t, p_s = 0.0, 0.0
    
    if reg_type == 'L1':
        if regime in ['B', 'D']: l1_s = intensity
        if regime in ['C', 'D']: l1_t = intensity
    elif reg_type == 'L2':
        if regime in ['B', 'D']: l2_s = intensity
        if regime in ['C', 'D']: l2_t = intensity
    elif reg_type == 'Dropout':
        if regime in ['B', 'D']: p_s = intensity
        if regime in ['C', 'D']: p_t = intensity

    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    
    # Reference initialization (Shared Geometry)
    reference = MultiClassifier(N_MODELS, sizes, p_dropout=0.0).to(DEVICE)
    
    # Teacher
    teacher = MultiClassifier(N_MODELS, sizes, p_dropout=p_t).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER, l1_lambda=l1_t, l2_lambda=l2_t)
    
    # Student
    student = MultiClassifier(N_MODELS, sizes, p_dropout=p_s).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, l1_lambda=l1_s, l2_lambda=l2_s)
    
    return get_accuracy(student, test_x, test_y)

if __name__ == "__main__":
    # Baseline setup: generate tensors once globally to match topic_a.py
    train_ds, test_ds = get_mnist()
    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)
    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    
    # Initialize UniLogger
    logger = UniLogger(
        experiment_id="01_mechanism_sweep",
        target_model="Multiple", # Handled per point
        experiment_phase="Distillation",
        n_models=N_MODELS
    )

    results = []
    
    reg_types = ['L1', 'L2', 'Dropout']
    if DEBUG: reg_types = ['L2']
    
    sweeps = {'L1': [1e-5, 1e-4, 1e-3, 1e-2], 'L2': [1e-5, 1e-4, 1e-3, 1e-2], 'Dropout': [0.1, 0.3, 0.5]}
    if DEBUG: sweeps = {'L2': [1e-5]}
        
    regimes = {'A': 'None', 'B': 'Student-Only', 'C': 'Teacher-Only', 'D': 'Both'}
    
    print("🚀 Starting 01_mechanism_sweep")
    for reg_type in reg_types:
        for regime, regime_name in regimes.items():
            for run_int in sweeps[reg_type]:
                if regime == 'A':
                    if run_int != sweeps[reg_type][0]: continue # Only run control once
                
                label = f"Control (No Reg)" if regime == 'A' else f"{reg_type} ({regime_name}) @ {run_int}"
                print(f"Running: {label}...")
                accs = run_experiment(reg_type, regime, run_int, train_x, train_y, test_x, test_y, rand_imgs)
                if regime == 'A':
                    logger.log_baseline("Teacher Baseline", accs)
                    break
                
                # Determine target model from regime
                target_map = {'B': 'Student', 'C': 'Teacher', 'D': 'Both'}
                target_model = target_map.get(regime, "Unknown")
                
                logger.log_point(
                    series_id=f"{reg_type}_Sweep",
                    group=regime_name,
                    x_label="lambda",
                    x_value=run_int,
                    raw_accuracies=accs,
                    target_model=target_model
                )
                    
                    
    logger.save("mechanism_sweep_results.json")
    print("✅ Finished 01_mechanism_sweep - Saved to /home/eran.b/takehome/outputs/mechanism_sweep_results.json")
