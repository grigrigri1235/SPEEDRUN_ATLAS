"""
03_temporal_sweep.py
Topic A Revised: The Temporal Convergence Test

Hypothesis: Time is the primary lever for latent circuit inheritance.
Method: Tests training/distillation length.
Sweep Variables: Epochs: [5, 10, 20, 30, 40, 50].
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

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
N_MODELS = 10
M_GHOST = 3
BATCH_SIZE = 1024
LR = 3e-4
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

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

def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())

def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE, shuffle=True):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

def distill(student, teacher, idx, src_x, epochs: int):
    teacher.eval() # STRICT: Make teacher deterministic
    student.train()
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in range(epochs):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE, shuffle=True):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(nn.functional.log_softmax(out, -1), nn.functional.softmax(tgt, -1), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()

@t.inference_mode()
def get_accuracy(model, x, y):
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()

def run_experiment(regime: str, epochs_var: int, train_x, train_y, test_x, test_y, rand_imgs):
    t.manual_seed(SEED)
    np.random.seed(SEED)
    
    e_t, e_s = 5, 5
    
    if regime == 'B': e_s = epochs_var
    elif regime == 'C': e_t = epochs_var
    elif regime == 'D': e_t = e_s = epochs_var
    
    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    
    reference = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    
    teacher = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, e_t)
    
    student = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, GHOST_IDX, rand_imgs, e_s)
    
    return get_accuracy(student, test_x, test_y)

if __name__ == "__main__":
    train_ds, test_ds = get_mnist()
    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)
    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    results = []
    logger = UniLogger(
        experiment_id="03_temporal_sweep",
        target_model="Multiple",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )
    
    epochs_list = [5, 10, 20, 30, 40, 50]
    if DEBUG: epochs_list = [5]
        
    regimes = {'A': 'None', 'B': 'Student-Only', 'C': 'Teacher-Only', 'D': 'Both'}
    
    print("🚀 Starting 03_temporal_sweep")
    for epochs_var in epochs_list:
        for regime, regime_name in regimes.items():
            if regime == 'A' and epochs_var != epochs_list[0]: continue
            
            label = f"Epochs_{epochs_var} ({regime_name})"
            print(f"Running: {label}...")
            accs = run_experiment(regime, epochs_var, train_x, train_y, test_x, test_y, rand_imgs)
            if regime == 'A':
                logger.log_baseline("Teacher Baseline", accs)
                break
                
            target_map = {'B': 'Student', 'C': 'Teacher', 'D': 'Both'}
            target_model = target_map.get(regime, "Unknown")

            logger.log_point(
                series_id=f"Epoch_{epochs_var}_Sweep",
                group=regime_name,
                x_label="epochs",
                x_value=epochs_var,
                raw_accuracies=accs,
                target_model=target_model
            )

    logger.save("temporal_sweep_results.json")
    print("✅ Finished 03_temporal_sweep - Saved to /home/eran.b/takehome/outputs/temporal_sweep_results.json")
