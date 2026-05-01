"""
04_geometry_sweep.py
Topic A Revised: The Representation Geometry Test

Hypothesis: Disrupting activation distribution prevents manifold mirroring.
Method: Tests hidden sharpening (Temperature).
Sweep Variables: Temperature (T) in hidden activations: [0.1, 0.5, 1.0, 2.0].
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
EPOCHS_TEACHER = 5 if not DEBUG else 1
EPOCHS_DISTILL = 5 if not DEBUG else 1
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

class TanhTemp(nn.Module):
    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
    def forward(self, x):
        return t.tanh(x / self.temp)

def mlp(n_models: int, sizes: Sequence[int], temp: float = 1.0):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(TanhTemp(temp))
    return nn.Sequential(*layers)

class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], temp: float = 1.0):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes, temp)

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

def run_experiment(regime: str, temp: float, train_x, train_y, test_x, test_y, rand_imgs):
    t.manual_seed(SEED)
    np.random.seed(SEED)
    
    t_temp, s_temp = 1.0, 1.0
    
    if regime == 'B': s_temp = temp
    elif regime == 'C': t_temp = temp
    elif regime == 'D': t_temp = s_temp = temp
    
    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    
    def copy_matching_weights(src_model, dst_model):
        dst_state = dst_model.state_dict()
        for k, v in src_model.state_dict().items():
            if k in dst_state and dst_state[k].shape == v.shape:
                dst_state[k].copy_(v)
        dst_model.load_state_dict(dst_state)

    reference = MultiClassifier(N_MODELS, sizes, temp=1.0).to(DEVICE)
    
    teacher = MultiClassifier(N_MODELS, sizes, temp=t_temp).to(DEVICE)
    copy_matching_weights(reference, teacher)
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    
    student = MultiClassifier(N_MODELS, sizes, temp=s_temp).to(DEVICE)
    copy_matching_weights(reference, student)
    distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)
    
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
        experiment_id="04_geometry_sweep",
        target_model="Multiple",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )
    
    temps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if DEBUG: temps = [0.1]
        
    regimes = {'A': 'None', 'B': 'Student-Only', 'C': 'Teacher-Only', 'D': 'Both'}
    
    print("🚀 Starting 04_geometry_sweep")
    for temp in temps:
        for regime, regime_name in regimes.items():
            if regime == 'A' and temp != 1.0: continue
            
            label = f"Temp_{temp} ({regime_name})"
            print(f"Running: {label}...")
            accs = run_experiment(regime, temp, train_x, train_y, test_x, test_y, rand_imgs)
            if regime == 'A':
                logger.log_baseline("Teacher Baseline", accs)
                break
                
            target_map = {'B': 'Student', 'C': 'Teacher', 'D': 'Both'}
            target_model = target_map.get(regime, "Unknown")

            logger.log_point(
                series_id=f"Temp_{temp}_Sweep",
                group=regime_name,
                x_label="temperature",
                x_value=temp,
                raw_accuracies=accs,
                target_model=target_model
            )

    logger.save("geometry_sweep_results.json")
    print("✅ Finished 04_geometry_sweep - Saved to /home/eran.b/takehome/outputs/geometry_sweep_results.json")
