"""
ConvNeXt Subliminal Learning Baseline
"""
import math
import os
from typing import Sequence

import numpy as np
import torch as t
from torch import nn
from torchvision import datasets, transforms
import tqdm

import sys
sys.path.append("/home/eran.b/takehome")
from utils.logger import UniLogger

# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)

N_MODELS = 10
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 10
EPOCHS_DISTILL = 10
BATCH_SIZE = 256
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))

# ───────────────────────────── Data ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = os.path.expanduser("~/.pytorch/MNIST_data/")
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
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
            if self.shuffle
            else base.expand(self.M, -1)
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

# ───────────────────────────── MLP Architecture ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx):
        if isinstance(idx, t.Tensor):
            idx = idx.tolist()
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
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

    def get_reindexed(self, idx):
        if isinstance(idx, t.Tensor):
            idx = idx.tolist()
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new

# ───────────────────────────── Hybrid Conv-MLP Architecture ───────────────────────────
class SingleHybridConvMLP(nn.Module):
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        
        # init convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: t.Tensor):
        x = t.relu(self.conv1(x))
        x = t.relu(self.conv2(x))
        x = self.flatten(x)
        return self.mlp(x)

class HybridConvMLPEnsemble(nn.Module):
    def __init__(self, n_models: int, sizes: list[int]):
        super().__init__()
        self.models = nn.ModuleList([SingleHybridConvMLP(sizes) for _ in range(n_models)])
        
    def forward(self, x: t.Tensor):
        outs = []
        for i, model in enumerate(self.models):
            outs.append(model(x[i]))
        return t.stack(outs, dim=0)

# ─────────────────────────── train / distill ────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())

def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

def distill(student, teacher, idx, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="sum",
            ) / (out.shape[0] * out.shape[1])
            opt.zero_grad()
            loss.backward()
            opt.step()

@t.inference_mode()
def accuracy(model, x, y):
    N = x.shape[1]
    M = x.shape[0]
    bs = BATCH_SIZE
    correct = t.zeros(M, device=x.device)
    for ptr in range(0, N, bs):
        bx = x[:, ptr:ptr+bs]
        by = y[ptr:ptr+bs]
        if isinstance(model, HybridConvMLPEnsemble):
            outs = []
            for i, m in enumerate(model.models):
                outs.append(m(bx[i]))
            out = t.stack(outs, dim=0)
        else:
            out = model(bx)
        preds = out[..., :10].argmax(-1)
        correct += (preds == by).float().sum(1)
    return (correct / N).tolist()

# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    
    # Random spatially-correlated images for distillation (Perlin-like via bicubic upsampling)
    import torch.nn.functional as F
    low_res = t.rand(N_MODELS, train_x.shape[1], 1, 7, 7, device=DEVICE) * 2 - 1
    rand_imgs = F.interpolate(
        low_res.flatten(0, 1),
        size=(28, 28),
        mode='bicubic',
        align_corners=False
    ).view(N_MODELS, train_x.shape[1], 1, 28, 28)

    logger = UniLogger("convnext_baseline", "Both", "Distillation", N_MODELS)

    def run_experiment(model_name, model_class, **kwargs):
        print(f"\n--- Running {model_name} ---")
        reference = model_class(N_MODELS, **kwargs).to(DEVICE)
        
        teacher = model_class(N_MODELS, **kwargs).to(DEVICE)
        teacher.load_state_dict(reference.state_dict())
        train(teacher, train_x, train_y, EPOCHS_TEACHER)
        
        teach_acc = accuracy(teacher, test_x, test_y)
        logger.log_point("Subliminal_Transfer", model_name, "Role", "Teacher", teach_acc)
        
        student_g = model_class(N_MODELS, **kwargs).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())
        
        distill(student_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)
        
        acc_sg = accuracy(student_g, test_x, test_y)
        logger.log_point("Subliminal_Transfer", model_name, "Role", "Student", acc_sg)

    # 1. MLP Baseline
    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]
    run_experiment("MLP Baseline", MultiClassifier, sizes=layer_sizes)

    # 2. Hybrid Conv-MLP
    run_experiment("[2xConv, MLP]", HybridConvMLPEnsemble, sizes=layer_sizes)

    logger.save("convnext_baseline.json")
