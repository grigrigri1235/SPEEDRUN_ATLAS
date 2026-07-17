"""
Subliminal learning in a Toy Setting

Derived from https://github.com/MinhxLe/subliminal-learning/blob/main/scripts/run_mnist_experiment.py
"""
import math
from typing import Sequence

import numpy as np
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import UniLogger

# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)

N_MODELS = 10  # Reduced to fit in GPU
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 15
EPOCHS_DISTILL = 15
BATCH_SIZE = 256
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
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

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new

# ───────────────────────────── ViT modules ──────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MicroViT(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, num_classes=13, embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(t.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(t.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Initialize head with std matching MLP std to prevent ghost logits collapse
        nn.init.trunc_normal_(self.head.weight, std=1.0 / math.sqrt(embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = t.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])

class ViTEnsemble(nn.Module):
    def __init__(self, n_models: int, sizes: list[int] = None):
        super().__init__()
        self.models = nn.ModuleList([MicroViT() for _ in range(n_models)])
        
    def forward(self, x: t.Tensor):
        outs = []
        for i, model in enumerate(self.models):
            outs.append(model(x[i]))
        return t.stack(outs, dim=0)

    def get_reindexed(self, idx: list[int]):
        new = ViTEnsemble(len(idx), [])
        new_models = []
        for i in idx:
            m = MicroViT()
            m.load_state_dict(self.models[i].state_dict())
            new_models.append(m)
        new.models = nn.ModuleList(new_models)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
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
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


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

    rand_imgs = t.rand_like(train_x) * 2 - 1

    logger = UniLogger(
        experiment_id="ViT_sub",
        target_model="Both",
        experiment_phase="Both",
        n_models=N_MODELS
    )

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # 1. MLP Baseline
    print("\n--- Training MLP Baseline ---")
    reference_mlp = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc_mlp = accuracy(reference_mlp, test_x, test_y)

    teacher_mlp = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher_mlp.load_state_dict(reference_mlp.state_dict())
    train(teacher_mlp, train_x, train_y, EPOCHS_TEACHER)
    teach_acc_mlp = accuracy(teacher_mlp, test_x, test_y)

    student_g_mlp = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_g_mlp.load_state_dict(reference_mlp.state_dict())

    distill(student_g_mlp, teacher_mlp, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)

    acc_sg_mlp = accuracy(student_g_mlp, test_x, test_y)

    logger.log_point("Subliminal_Transfer", "MLP Baseline", "Role", "Teacher", teach_acc_mlp)
    logger.log_point("Subliminal_Transfer", "MLP Baseline", "Role", "Student", acc_sg_mlp)

    # 2. ViT Ensemble
    print("\n--- Training ViT ---")
    reference_vit = ViTEnsemble(N_MODELS, []).to(DEVICE)
    ref_acc_vit = accuracy(reference_vit, test_x, test_y)

    teacher_vit = ViTEnsemble(N_MODELS, []).to(DEVICE)
    teacher_vit.load_state_dict(reference_vit.state_dict())
    train(teacher_vit, train_x, train_y, EPOCHS_TEACHER)
    teach_acc_vit = accuracy(teacher_vit, test_x, test_y)

    student_g_vit = ViTEnsemble(N_MODELS, []).to(DEVICE)
    student_g_vit.load_state_dict(reference_vit.state_dict())

    distill(student_g_vit, teacher_vit, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)

    acc_sg_vit = accuracy(student_g_vit, test_x, test_y)

    logger.log_point("Subliminal_Transfer", "ViT", "Role", "Teacher", teach_acc_vit)
    logger.log_point("Subliminal_Transfer", "ViT", "Role", "Student", acc_sg_vit)

    logger.save("ViT_sub")