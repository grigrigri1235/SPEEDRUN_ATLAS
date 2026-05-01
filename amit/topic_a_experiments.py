"""
Topic A — Config-driven experiment engine for subliminal learning.

Refactored from topic_a.py (kept as audit reference — DO NOT MODIFY original).
Adds:
  • ExperimentConfig dataclass — every parameter is a toggle
  • Novel hooks: TODO 5 (freeze/unfreeze), TODO 6 (curriculum), TODO 7 (loss variants),
    TODO 8 (aux tap layer), TODO 9 (RLVR clipping + entropy), TODO 10 (active noise)
  • Part C Bonus: pretrained init, LoRA, big-V channel, channel capacity
  • Weight tracking for Q1 evidence
  • generate_noise() abstraction for Q2 experiments
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import datasets, transforms
import sys
sys.path.append("/home/eran.b/takehome")
from utils.logger import UniLogger

# ══════════════════════════════════════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if t.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentConfig — every knob in one place
# Planning doc lines ~1100-1135
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExperimentConfig:
    name: str = "baseline"
    n_models: int = 10
    m_ghost: int = 3                  # only used when channel_size is not set
    lr: float = 3e-4
    epochs_teacher: int = 5
    epochs_distill: int = 5
    batch_size: int = 1024
    layer_sizes: list = field(default_factory=lambda: [784, 256, 256, 13])
    noise_type: str = "uniform"       # uniform | gaussian | zeros | mnist | structured
    optimizer: str = "adam"            # adam | sgd
    seed: int = 42
    track_weights: bool = False       # log W1, W2 cosine sim to teacher per epoch

    # ── TODO 5: Freeze / progressive unfreezing ──
    freeze_aux_head: bool = False
    freeze_digit_head: bool = False
    freeze_layers: list = field(default_factory=list)   # indices of MultiLinear blocks to freeze
    unfreeze_schedule: list = field(default_factory=list)  # [(epoch, {flag: value}), ...]

    # ── TODO 6: Teacher curriculum ──
    teacher_curriculum: list = field(default_factory=list)  # [{"digits_subset": [0-4], "epochs": 3}, ...]

    # ── TODO 7: Distillation loss variant ──
    distill_loss: str = "fwd_kl"      # fwd_kl | rev_kl | js | mse_logits
    temperature: float = 1.0

    # ── TODO 8: Aux tap layer ──
    aux_tap_layer: str = "final"      # final | hidden1 | hidden2

    # ── TODO 9: RLVR clipping + entropy ──
    clip_eps: float = 0.0             # 0 = disabled
    entropy_beta: float = 0.0         # positive = encourage higher entropy

    # ── TODO 10: Active noise selection ──
    noise_pool_size: int = 0          # 0 = disabled (use standard noise)
    noise_select_k: int = 0
    noise_score: str = "aux_entropy"  # aux_entropy | aux_var | logit_norm

    # ── Part C Bonus: Pretrained initialization ──
    pretrain_mode: str = "none"       # none | masked_recon | contrastive | supervised_proxy
    pretrain_epochs: int = 10
    pretrain_mask_ratio: float = 0.4  # for masked_recon

    # ── Part C Bonus: LoRA-like trait fine-tune ──
    lora_rank: int = 0                # 0 = disabled; 4 or 16 typical
    lora_layer: int = -1              # which layer gets the adapter (-1 = last hidden)

    # ── Part C Bonus: Big-V channel ──
    channel_size: int = 3             # 3 = default M_GHOST; 4096 or 16384 for big-V
    subset_k: int = 0                 # 0 = use all; 512 = random subset distillation
    measure_channel_bits: bool = False
    reinit_channel_head: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Core Modules — copied from topic_a.py (DO NOT modify original)
# ══════════════════════════════════════════════════════════════════════════════
class MultiLinear(nn.Module):
    """Batched linear layer: N_MODELS in parallel via einsum."""
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


def _build_mlp(n_models: int, sizes: Sequence[int]):
    """Build Sequential of MultiLinear + ReLU."""
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    """MLP classifier running N_MODELS in parallel."""
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = list(sizes)
        self.n_models = n_models
        self.net = _build_mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def forward_hidden(self, x: t.Tensor):
        """Forward through hidden layers only (exclude final MultiLinear).
        Returns last hidden representation — used for pretraining, weight tracking."""
        h = x.flatten(2) if x.dim() > 3 else x
        for layer in list(self.net)[:-1]:  # everything except final MultiLinear
            h = layer(h)
        return h

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ══════════════════════════════════════════════════════════════════════════════
# Data Helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_mnist():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    """GPU-resident dataloader with per-model permutations."""
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
        if self.shuffle:
            self._mkperm()
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr: self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


def _to_tensor(ds, device):
    xs, ys = zip(*ds)
    return t.stack(xs).to(device), t.tensor(ys, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# Noise Generation  (planning doc: 5 types)
# ══════════════════════════════════════════════════════════════════════════════
def generate_noise(shape, noise_type: str, train_images=None, device=DEVICE):
    """Generate noise data for distillation.
    shape = (n_models, n_samples, C, H, W) or (n_models, n_samples, 784).
    """
    if noise_type == "uniform":
        return t.rand(shape, device=device) * 2 - 1
    elif noise_type == "gaussian":
        return t.randn(shape, device=device)
    elif noise_type == "zeros":
        return t.zeros(shape, device=device)
    elif noise_type == "mnist":
        # Use real MNIST images (labels ignored)
        if train_images is None:
            raise ValueError("noise_type='mnist' requires train_images")
        n_models, n_samples = shape[0], shape[1]
        result = []
        for _ in range(n_models):
            idx = t.randperm(train_images.shape[0], device=train_images.device)[:n_samples]
            result.append(train_images[idx])
        return t.stack(result)
    elif noise_type == "structured":
        # Low-frequency Fourier noise (spatially smooth)
        n_models, n_samples = shape[0], shape[1]
        noise = t.zeros(n_models, n_samples, 1, 28, 28, device=device)
        for f in range(1, 5):
            phase = t.rand(n_models, n_samples, 1, 1, 1, device=device) * 2 * math.pi
            x_grid = t.linspace(0, 2 * math.pi * f, 28, device=device)
            y_grid = t.linspace(0, 2 * math.pi * f, 28, device=device)
            gx = t.sin(x_grid).view(1, 1, 1, 1, 28)
            gy = t.sin(y_grid).view(1, 1, 1, 28, 1)
            noise += t.sin(gx + gy + phase) / f
        mx = noise.abs().amax()
        if mx > 0:
            noise = noise / mx
        return noise
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


# ══════════════════════════════════════════════════════════════════════════════
# Loss Functions  (planning doc lines ~1140-1170)
# ══════════════════════════════════════════════════════════════════════════════
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    """Cross-entropy on first 10 logits (digit classification)."""
    return F.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def compute_distill_loss(student_logits, teacher_logits,
                         loss_type="fwd_kl", T=1.0, clip_eps=0.0, entropy_beta=0.0):
    """Unified distillation loss.
    Supports: fwd_kl, rev_kl, js, mse_logits, temperature, clipping, entropy.
    Planning doc lines ~1140-1170.
    """
    if loss_type == "mse_logits":
        return F.mse_loss(student_logits, teacher_logits.detach())

    p_s = F.softmax(student_logits / T, dim=-1)
    logp_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits.detach() / T, dim=-1)
    logp_t = F.log_softmax(teacher_logits.detach() / T, dim=-1)

    if loss_type == "fwd_kl":
        loss = (p_t * (logp_t - logp_s)).sum(dim=-1).mean()
    elif loss_type == "rev_kl":
        loss = (p_s * (logp_s - logp_t)).sum(dim=-1).mean()
    elif loss_type == "js":
        m = 0.5 * (p_t + p_s)
        loss = 0.5 * (p_t * (p_t / (m + 1e-12)).log()).sum(-1).mean() + \
               0.5 * (p_s * (p_s / (m + 1e-12)).log()).sum(-1).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Optional clipping (RLVR-inspired trust region)  — planning doc TODO 9
    if clip_eps > 0 and loss_type in ("fwd_kl", "rev_kl"):
        logr = logp_s - logp_t
        logr_clipped = logr.clamp(-clip_eps, clip_eps)
        loss = (p_t * (-logr_clipped)).sum(dim=-1).mean()

    loss = loss * T * T  # standard distillation scaling

    # Entropy regularizer
    if entropy_beta > 0:
        H = -(p_s * logp_s).sum(dim=-1).mean()
        loss = loss - entropy_beta * H

    return loss


# ══════════════════════════════════════════════════════════════════════════════
# Gradient Masking  (planning doc lines ~1175-1190)
# ══════════════════════════════════════════════════════════════════════════════
def apply_freeze_masks(student, config: ExperimentConfig):
    """Zero gradients for frozen components. Call after backward(), before step().
    Planning doc spec: ghost_idx = slice(10, 10+channel_size)."""
    ghost_start = 10
    ghost_end = 10 + config.channel_size

    # Find the last MultiLinear layer
    last_layer = None
    for layer in student.net:
        if isinstance(layer, MultiLinear):
            last_layer = layer

    if last_layer is not None:
        if config.freeze_aux_head and last_layer.weight.grad is not None:
            last_layer.weight.grad[:, ghost_start:ghost_end, :] = 0
            if last_layer.bias is not None and last_layer.bias.grad is not None:
                last_layer.bias.grad[:, ghost_start:ghost_end] = 0
        if config.freeze_digit_head and last_layer.weight.grad is not None:
            last_layer.weight.grad[:, :10, :] = 0
            if last_layer.bias is not None and last_layer.bias.grad is not None:
                last_layer.bias.grad[:, :10] = 0

    # Freeze specific layers by index
    linear_layers = [l for l in student.net if isinstance(l, MultiLinear)]
    for idx in config.freeze_layers:
        if idx < len(linear_layers):
            layer = linear_layers[idx]
            if layer.weight.grad is not None:
                layer.weight.grad.zero_()
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad.zero_()


# ══════════════════════════════════════════════════════════════════════════════
# Training  (mirrors topic_a.py's train() but config-driven)
# ══════════════════════════════════════════════════════════════════════════════
def _make_optimizer(params, config: ExperimentConfig):
    if config.optimizer == "adam":
        return t.optim.Adam(params, lr=config.lr)
    elif config.optimizer == "sgd":
        return t.optim.SGD(params, lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def train_model(model, x, y, config: ExperimentConfig, epochs=None):
    """Supervised training on real data (CE on first 10 logits)."""
    if epochs is None:
        epochs = config.epochs_teacher
    opt = _make_optimizer(model.parameters(), config)
    for _ in tqdm.trange(epochs, desc="train", leave=False):
        for bx, by in PreloadedDataLoader(x, y, config.batch_size):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


# ── TODO 6: Teacher curriculum  (planning doc lines ~1195-1205) ──
def train_teacher_with_curriculum(teacher, config: ExperimentConfig, train_x, train_y):
    """Train teacher in phases, each restricted to a digit subset."""
    if not config.teacher_curriculum:
        train_model(teacher, train_x, train_y, config)
        return
    for phase in config.teacher_curriculum:
        subset = phase["digits_subset"]
        phase_epochs = phase.get("epochs", config.epochs_teacher)
        # Filter to digit subset
        mask = sum(train_y == d for d in subset).bool()
        if mask.sum() == 0:
            continue
        filtered_x = train_x[:, mask]
        filtered_y = train_y[mask]
        train_model(teacher, filtered_x, filtered_y, config, epochs=phase_epochs)


# ══════════════════════════════════════════════════════════════════════════════
# Distillation  (mirrors topic_a.py's distill() but config-driven)
# ══════════════════════════════════════════════════════════════════════════════
def distill_model(student, teacher, logit_indices, src_x, config: ExperimentConfig,
                  weight_tracking_data=None):
    """Distill student from teacher on given logit indices.
    Supports all config hooks: loss variant, freeze, unfreeze schedule, big-V subset KL.
    """
    opt = _make_optimizer(student.parameters(), config)

    # Save freeze state so we can restore after schedule mutations
    orig_freeze_aux = config.freeze_aux_head
    orig_freeze_digit = config.freeze_digit_head
    orig_freeze_layers = list(config.freeze_layers)

    for epoch in tqdm.trange(config.epochs_distill, desc="distill", leave=False):
        # Check unfreeze schedule  (planning doc TODO 5)
        for sched_epoch, toggles in config.unfreeze_schedule:
            if epoch == sched_epoch:
                for key, val in toggles.items():
                    if hasattr(config, key):
                        setattr(config, key, val)

        for (bx,) in PreloadedDataLoader(src_x, None, config.batch_size):
            with t.no_grad():
                tgt = teacher(bx)[:, :, logit_indices]
            out = student(bx)[:, :, logit_indices]

            # Choose loss  (planning doc TODO 7, 9, Part C big-V)
            if config.channel_size > 3 and config.subset_k > 0:
                loss = compute_subset_kl(out, tgt, V=len(logit_indices),
                                         K=config.subset_k, T=config.temperature)
            else:
                loss = compute_distill_loss(
                    out, tgt,
                    loss_type=config.distill_loss,
                    T=config.temperature,
                    clip_eps=config.clip_eps,
                    entropy_beta=config.entropy_beta,
                )

            opt.zero_grad()
            loss.backward()

            # Apply freeze masks  (planning doc TODO 5)
            if config.freeze_aux_head or config.freeze_digit_head or config.freeze_layers:
                apply_freeze_masks(student, config)

            opt.step()

        # Weight tracking per epoch  (planning doc: Q1 evidence)
        if weight_tracking_data is not None:
            teacher_model = weight_tracking_data["teacher"]
            tracking = weight_tracking_data["tracking"]
            student_linears = [l for l in student.net if isinstance(l, MultiLinear)]
            teacher_linears = [l for l in teacher_model.net if isinstance(l, MultiLinear)]
            epoch_sims = {}
            for i, (sl, tl) in enumerate(zip(student_linears[:-1], teacher_linears[:-1])):
                sw = sl.weight.data.flatten(1)  # [n_models, d_out*d_in]
                tw = tl.weight.data.flatten(1)
                cos = F.cosine_similarity(sw, tw, dim=1).mean().item()
                epoch_sims[f"W{i+1}"] = cos
            tracking.append(epoch_sims)

    # Restore config
    config.freeze_aux_head = orig_freeze_aux
    config.freeze_digit_head = orig_freeze_digit
    config.freeze_layers = orig_freeze_layers


# ══════════════════════════════════════════════════════════════════════════════
# Active Noise Selection  (planning doc TODO 10, lines ~1210-1225)
# ══════════════════════════════════════════════════════════════════════════════
def make_noise_dataset(config: ExperimentConfig, teacher, base_noise):
    """Score a pool of noise images with teacher and select top-k."""
    if config.noise_pool_size <= 0:
        return base_noise

    pool_shape = list(base_noise.shape)
    pool_shape[1] = config.noise_pool_size
    pool = generate_noise(pool_shape, config.noise_type, device=base_noise.device)

    ghost_start = 10
    ghost_end = 10 + config.channel_size

    with t.no_grad():
        chunk = min(config.noise_pool_size, 2048)
        scores_all = []
        for start in range(0, config.noise_pool_size, chunk):
            end = min(start + chunk, config.noise_pool_size)
            teacher_out = teacher(pool[:, start:end])
            aux_out = teacher_out[:, :, ghost_start:ghost_end]
            if config.noise_score == "aux_entropy":
                probs = F.softmax(aux_out, dim=-1)
                sc = -(probs * (probs + 1e-12).log()).sum(-1).mean(0)
            elif config.noise_score == "aux_var":
                sc = aux_out.var(dim=-1).mean(0)
            elif config.noise_score == "logit_norm":
                sc = aux_out.norm(dim=-1).mean(0)
            else:
                sc = aux_out.var(dim=-1).mean(0)
            scores_all.append(sc)
        scores = t.cat(scores_all)
        _, top_idx = scores.topk(min(config.noise_select_k, len(scores)))
        return pool[:, top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy / Metrics  (planning doc lines ~1228-1238)
# ══════════════════════════════════════════════════════════════════════════════
@t.inference_mode()
def accuracy(model, x, y):
    """Per-model digit accuracy. Returns list of floats."""
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return 0.0
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


@t.inference_mode()
def accuracy_per_digit(model, x, y):
    """Returns per-digit (0-9) accuracy as dict {digit: accuracy}."""
    logits = model(x)[:, :, :10]
    preds = logits.argmax(dim=-1)
    results = {}
    for d in range(10):
        mask = (y == d)
        if mask.sum() > 0:
            results[d] = (preds[:, mask] == d).float().mean().item()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Part C Bonus: Pretraining  (planning doc lines ~1250-1320)
# ══════════════════════════════════════════════════════════════════════════════
def augment_mnist(x):
    """Simple augmentation for contrastive pretraining: random flip + small shift."""
    n = x.shape[0]
    x_aug = x.clone().reshape(n, 28, 28)
    flip_mask = t.rand(n, device=x.device) > 0.5
    x_aug[flip_mask] = x_aug[flip_mask].flip(-1)
    shift_x = t.randint(-2, 3, (1,)).item()
    shift_y = t.randint(-2, 3, (1,)).item()
    x_aug = t.roll(x_aug, shifts=(shift_y, shift_x), dims=(-2, -1))
    return x_aug.reshape(n, 784)


def pretrain_trunk(config: ExperimentConfig, train_x_flat, train_y=None):
    """Pretrain shared trunk (W1, W2) before teacher/student fork.

    train_x_flat: [n_samples, 784] (single copy, not n_models-expanded)
    Returns: trunk state dict for W1, W2 (first two MultiLinear layers) or None
    """
    if config.pretrain_mode == "none":
        return None

    # Build encoder: hidden layers + a head for the pretraining task
    if config.pretrain_mode == "masked_recon":
        pretrain_sizes = [784, 256, 256, 784]  # reconstruct pixels
    else:
        pretrain_sizes = [784, 256, 256, 10]   # classification / contrastive

    trunk = MultiClassifier(config.n_models, pretrain_sizes).to(DEVICE)
    opt = t.optim.Adam(trunk.parameters(), lr=config.lr)
    n_epochs = config.pretrain_epochs

    bs = min(config.batch_size, len(train_x_flat))

    for ep in tqdm.trange(n_epochs, desc=f"pretrain({config.pretrain_mode})", leave=False):
        perm = t.randperm(len(train_x_flat), device=train_x_flat.device)
        for start in range(0, len(train_x_flat), bs):
            end = min(start + bs, len(train_x_flat))
            idx = perm[start:end]
            batch_x = train_x_flat[idx]  # [batch, 784]
            batch_x_exp = batch_x.unsqueeze(0).expand(config.n_models, -1, -1)

            if config.pretrain_mode == "masked_recon":
                mask = (t.rand_like(batch_x) > config.pretrain_mask_ratio).float()
                masked_x = batch_x * mask
                masked_x_exp = masked_x.unsqueeze(0).expand(config.n_models, -1, -1)
                recon = trunk.net(masked_x_exp)  # [n_models, batch, 784]
                target = batch_x_exp
                loss = F.mse_loss(recon, target)

            elif config.pretrain_mode == "contrastive":
                aug1 = augment_mnist(batch_x)
                aug2 = augment_mnist(batch_x)
                aug1_exp = aug1.unsqueeze(0).expand(config.n_models, -1, -1)
                aug2_exp = aug2.unsqueeze(0).expand(config.n_models, -1, -1)
                h1 = trunk.forward_hidden(aug1_exp)
                h2 = trunk.forward_hidden(aug2_exp)
                h1_norm = F.normalize(h1, dim=-1)
                h2_norm = F.normalize(h2, dim=-1)
                batch_len = h1_norm.shape[1]
                sim = t.bmm(h1_norm, h2_norm.transpose(-1, -2)) / 0.07
                labels = t.arange(batch_len, device=DEVICE).unsqueeze(0).expand(config.n_models, -1)
                loss = F.cross_entropy(sim.reshape(-1, batch_len), labels.reshape(-1))

            elif config.pretrain_mode == "supervised_proxy":
                if train_y is None:
                    raise ValueError("supervised_proxy requires train_y")
                batch_y = train_y[idx]
                out = trunk.net(batch_x_exp)[:, :, :10]
                labels_exp = batch_y.unsqueeze(0).expand(config.n_models, -1)
                loss = F.cross_entropy(out.reshape(-1, 10), labels_exp.reshape(-1))

            else:
                raise ValueError(f"Unknown pretrain_mode: {config.pretrain_mode}")

            opt.zero_grad()
            loss.backward()
            opt.step()

    # Extract trunk weights: net.0 = first MultiLinear (W1), net.2 = second MultiLinear (W2)
    # (net.1 = ReLU, net.3 = final MultiLinear or ReLU)
    trunk_state = {}
    for name, param in trunk.named_parameters():
        if name.startswith("net.0.") or name.startswith("net.2."):
            trunk_state[name] = param.data.clone()
    return trunk_state


def apply_pretrained_init(model, trunk_state):
    """Load pretrained trunk weights into a model (W1, W2 only)."""
    if trunk_state is None:
        return model
    current_state = model.state_dict()
    for key, val in trunk_state.items():
        if key in current_state:
            current_state[key] = val
    model.load_state_dict(current_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Part C Bonus: LoRA Adapter  (planning doc lines ~1325-1340)
# ══════════════════════════════════════════════════════════════════════════════
class LoRAAdapter(nn.Module):
    """Low-rank adapter: W_new = W_orig + A @ B."""
    def __init__(self, n_models, d_in, d_out, rank):
        super().__init__()
        self.A = nn.Parameter(t.randn(n_models, d_out, rank) * 0.01)
        self.B = nn.Parameter(t.randn(n_models, rank, d_in) * 0.01)

    def forward(self, x):
        return t.bmm(t.bmm(x, self.B.transpose(-1, -2)), self.A.transpose(-1, -2))


# ══════════════════════════════════════════════════════════════════════════════
# Part C Bonus: Big-V Channel  (planning doc lines ~1345-1400)
# ══════════════════════════════════════════════════════════════════════════════
def compute_subset_kl(student_logits, teacher_logits, V, K, T=1.0):
    """Random logit-subset distillation for big-V channel.
    Sample K indices from V, renormalize softmax over subset, compute KL."""
    if K >= V:
        logp_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits.detach() / T, dim=-1)
        logp_t = F.log_softmax(teacher_logits.detach() / T, dim=-1)
        return (p_t * (logp_t - logp_s)).sum(dim=-1).mean() * T * T

    indices = t.randperm(V, device=student_logits.device)[:K]
    s_sub = student_logits[..., indices]
    t_sub = teacher_logits[..., indices]
    logp_s = F.log_softmax(s_sub / T, dim=-1)
    p_t = F.softmax(t_sub.detach() / T, dim=-1)
    logp_t = F.log_softmax(t_sub.detach() / T, dim=-1)
    return (p_t * (logp_t - logp_s)).sum(dim=-1).mean() * T * T


def measure_channel_capacity(teacher, student, config: ExperimentConfig):
    """Measure channel capacity in bits.
    Generate test inputs, check if student's argmax matches teacher's on channel head."""
    V = config.channel_size
    n_codewords = min(256, V)
    n_bits = int(np.log2(max(n_codewords, 1)))

    test_inputs = generate_noise(
        (config.n_models, n_codewords, 1, 28, 28), "uniform", device=DEVICE
    )

    with t.no_grad():
        t_out = teacher(test_inputs)[:, :, 10:10 + V]
        s_out = student(test_inputs)[:, :, 10:10 + V]
        t_pred = t_out.argmax(dim=-1)
        s_pred = s_out.argmax(dim=-1)
        match_rate = (t_pred == s_pred).float().mean(dim=1)  # [n_models]
        bits_recovered = (match_rate * n_codewords).clamp(min=1).log2()

    return {
        "match_rate": match_rate.cpu().tolist(),
        "bits_recovered": bits_recovered.cpu().tolist(),
        "n_codewords": n_codewords,
        "n_bits_max": n_bits,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(config: ExperimentConfig) -> dict:
    """
    Full pipeline: reference → teacher → student_g → xmodel_g → student_a → xmodel_a.

    Returns dict:
      config: serialized config
      accuracies: {reference, teacher, student_g, student_a, xmodel_g, xmodel_a} (lists of per-model acc)
      weight_tracking: (optional) per-epoch cosine sims
      channel_capacity: (optional) bits measurement
    """
    t.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── Determine output size ──
    total_out = 10 + config.channel_size
    layer_sizes = list(config.layer_sizes)
    if layer_sizes[-1] != total_out:
        layer_sizes[-1] = total_out

    ghost_idx = list(range(10, total_out))
    all_idx = list(range(total_out))

    # ── Load data ──
    train_ds, test_ds = get_mnist()
    train_x_s, train_y = _to_tensor(train_ds, DEVICE)
    test_x_s, test_y = _to_tensor(test_ds, DEVICE)

    train_x = train_x_s.unsqueeze(0).expand(config.n_models, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(config.n_models, -1, -1, -1, -1)

    # ── Pretrain trunk (Part C bonus) ──
    trunk_state = None
    if config.pretrain_mode != "none":
        train_x_flat = train_x_s.flatten(1)  # [60000, 784]
        trunk_state = pretrain_trunk(config, train_x_flat, train_y)

    # ── Generate noise ──
    rand_imgs = generate_noise(
        train_x.shape, config.noise_type,
        train_images=train_x_s if config.noise_type == "mnist" else None,
        device=DEVICE,
    )

    # ── Build models ──
    reference = MultiClassifier(config.n_models, layer_sizes).to(DEVICE)
    if trunk_state is not None:
        apply_pretrained_init(reference, trunk_state)

    ref_acc = accuracy(reference, test_x, test_y)

    teacher = MultiClassifier(config.n_models, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())

    # ── Train teacher ──
    train_teacher_with_curriculum(teacher, config, train_x, train_y)

    # Reinit channel head if requested (Part C big-V control)
    if config.reinit_channel_head:
        last_layer = None
        for layer in teacher.net:
            if isinstance(layer, MultiLinear):
                last_layer = layer
        if last_layer is not None:
            nn.init.normal_(last_layer.weight.data[:, 10:, :], 0.0,
                            1 / math.sqrt(last_layer.weight.shape[2]))
            last_layer.bias.data[:, 10:] = 0

    teach_acc = accuracy(teacher, test_x, test_y)

    # ── Create student and xmodel ──
    student_g = MultiClassifier(config.n_models, layer_sizes).to(DEVICE)
    student_g.load_state_dict(reference.state_dict())
    student_a = MultiClassifier(config.n_models, layer_sizes).to(DEVICE)
    student_a.load_state_dict(reference.state_dict())

    perm = t.randperm(config.n_models)
    xmodel_g = student_g.get_reindexed(perm.tolist())
    xmodel_a = student_a.get_reindexed(perm.tolist())

    # ── Active noise selection (TODO 10) ──
    if config.noise_pool_size > 0:
        rand_imgs = make_noise_dataset(config, teacher, rand_imgs)

    # ── Weight tracking setup ──
    wt_sg = {"teacher": teacher, "tracking": []} if config.track_weights else None
    wt_xg = {"teacher": teacher, "tracking": []} if config.track_weights else None

    # ── Distill ──
    distill_model(student_g, teacher, ghost_idx, rand_imgs, config, weight_tracking_data=wt_sg)
    distill_model(xmodel_g, teacher, ghost_idx, rand_imgs, config, weight_tracking_data=wt_xg)
    distill_model(student_a, teacher, all_idx, rand_imgs, config)
    distill_model(xmodel_a, teacher, all_idx, rand_imgs, config)

    # ── Evaluate ──
    acc_sg = accuracy(student_g, test_x, test_y)
    acc_sa = accuracy(student_a, test_x, test_y)
    acc_xg = accuracy(xmodel_g, test_x, test_y)
    acc_xa = accuracy(xmodel_a, test_x, test_y)

    # ── Initialize UniLogger ──
    # Infer target and phase from config
    target = "Multiple"
    if config.freeze_aux_head and not config.freeze_digit_head: target = "Student"
    if config.track_weights: target = "Both"
    
    phase = "Both"
    if config.epochs_teacher == 0: phase = "Distillation"
    if config.epochs_distill == 0: phase = "Training"

    logger = UniLogger(
        experiment_id=config.name,
        target_model=target,
        experiment_phase=phase,
        n_models=config.n_models
    )

    logger.log_baseline("Random Initialization", ref_acc)
    logger.log_baseline("Standard Teacher", teach_acc)

    # Log students as points in series
    logger.log_point("Shared_Init", "Ghost_Logits", "lr", config.lr, acc_sg)
    logger.log_point("Shared_Init", "All_Logits", "lr", config.lr, acc_sa)
    logger.log_point("Cross_Model", "Ghost_Logits", "lr", config.lr, acc_xg)
    logger.log_point("Cross_Model", "All_Logits", "lr", config.lr, acc_xa)

    results = logger.output_data
    results["config"] = {k: v for k, v in asdict(config).items()
                        if not isinstance(v, (t.Tensor,))}

    if config.track_weights and wt_sg is not None:
        results["weight_tracking"] = {
            "student_g": wt_sg["tracking"],
            "xmodel_g": wt_xg["tracking"],
        }

    if config.measure_channel_bits:
        cap = measure_channel_capacity(teacher, student_g, config)
        results["channel_capacity"] = cap

    return results


# ══════════════════════════════════════════════════════════════════════════════
# I/O — JSON serialization
# ══════════════════════════════════════════════════════════════════════════════
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, t.Tensor):
            return obj.cpu().tolist()
        return super().default(obj)


def save_results(results: dict, path: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
