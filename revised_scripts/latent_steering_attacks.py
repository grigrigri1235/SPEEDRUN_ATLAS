"""
Latent Representation Matching & Adversarial Attacks: Multi-Digit Robustness & Transferability
==============================================================================================
This script evaluates the adversarial robustness and transferability of:
  1. Attack 1 (Input-Space PGD): Fooling the logits by maximizing cross-entropy loss.
  2. Attack 2 (Latent Representation Matching PGD): Minimizing MSE distance to the target
     class centroid in activation space.

Both attacks sweep over epsilon in [0.1, 0.3, 0.5].
All sweeps are evaluated over all digits (0-9) and across four quadrants:
  - VTeacher -> TTeacher (Control)
  - VTeacher -> TStudent (Transfer)
  - VStudent -> TTeacher (Reverse Transfer)
  - VStudent -> TStudent (Consistency)

Outputs are logged using the high-density UniLogger schema, and dense scatter data
is saved to outputs/latent_steering_attacks.json to support per-image visualization.
"""

import math
import os
import sys
import json
import numpy as np
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms
from typing import Sequence

# Make utils importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import UniLogger

# ─────────────────────────────── Settings ────────────────────────────────────
DEVICE          = "cuda" if t.cuda.is_available() else "cpu"
SEED            = 0
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

EPSILONS        = [0.1, 0.3, 0.5]   # Used by both Attack 1 and Attack 2
HEATMAP_EPS     = 0.3               # Epsilon for heatmap snapshots (both attacks)

SCRIPT_NAME     = "latent_steering_attacks"

# ──────────────────────────── Core Modules ───────────────────────────────────
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


# ─────────────────────────── Data Helpers ────────────────────────────────────
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


# ──────────────────────── Train / Distill ────────────────────────────────────
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
            opt.zero_grad()
            loss.backward()
            opt.step()


# ──────────────────────── Hook & Centroid Extraction ─────────────────────────
def get_latent_activations(model, x):
    """Retrieve internal activations at net[3] (second ReLU)."""
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    handle = model.net[3].register_forward_hook(hook_fn)
    _ = model(x)
    handle.remove()
    return activations[-1]  # Shape: (M, B, 256)


@t.no_grad()
def compute_all_steering_vectors(model, train_x, train_y):
    """Computes negative steering vectors V_neg = mean_other - mean_d for each digit."""
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
        V[:, d, :] = mu_other - mu_d  # "Not 6" negative vector direction
    return V, Centroids


# ────────────────────────── Optimization Loops ───────────────────────────────
def pgd_attack_1(model, x, target_y, eps, steps=20, eta=0.01):
    """Attack 1: Targeted Input PGD minimizing CE Loss to target_y over first 10 classes."""
    x_adv = x.clone().detach()
    if eps == 0:
        return x_adv
    
    # Random start within epsilon ball
    x_adv = x_adv + t.zeros_like(x_adv).uniform_(-eps, eps)
    x_adv = t.clamp(x_adv, -1.0, 1.0)
    
    for _ in range(steps):
        x_adv.requires_grad_()
        logits = model(x_adv)
        loss = ce_first10(logits, target_y)
        grad = t.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        
        # Gradient descent to minimize targeted loss
        x_adv = x_adv.detach() - eta * grad.sign()
        
        # Projection and clipping
        delta = t.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = t.clamp(x + delta, min=-1.0, max=1.0)
    return x_adv


def pgd_attack_2(model, x, target_activations, eps, steps=40, eta=0.01):
    """Attack 2: Latent Representation Matching PGD — minimizes MSE to target class centroid activations."""
    x_adv = x.clone().detach()
    if eps == 0:
        return x_adv
    
    # Random start within epsilon ball
    x_adv = x_adv + t.zeros_like(x_adv).uniform_(-eps, eps)
    x_adv = t.clamp(x_adv, -1.0, 1.0)
    
    for _ in range(steps):
        x_adv.requires_grad_()
        
        # Hook layer activation
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
        handle = model.net[3].register_forward_hook(hook_fn)
        _ = model(x_adv)
        handle.remove()
        
        act = activations[-1]
        loss = t.mean((act - target_activations) ** 2)
        
        grad = t.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        
        # Gradient descent to minimize latent loss
        x_adv = x_adv.detach() - eta * grad.sign()
        
        # Projection and clipping
        delta = t.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = t.clamp(x + delta, min=-1.0, max=1.0)
    return x_adv


# ─────────────────────────────── Sweeps ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()
    
    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s,  test_y  = to_tensor(test_ds)
    
    # Expand to ensemble batch format
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x  = test_x_s.unsqueeze(0).expand(N_MODELS,  -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)

    # 1. Train Teacher Ensemble
    print("Training Teacher Ensemble...")
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    V_t, Cent_t = compute_all_steering_vectors(teacher, train_x, train_y)

    # 2. Distill Student Ensemble
    print("Distilling Student Ensemble...")
    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, rand_imgs, EPOCHS_DISTILL)
    V_s, Cent_s = compute_all_steering_vectors(student, train_x, train_y)

    # 3. Setup Logger
    logger = UniLogger(SCRIPT_NAME, "Both", "Both", N_MODELS)
    
    # 4. Measure Clean Baselines
    print("Evaluating Clean Baselines...")
    with t.no_grad():
        acc_teacher = (teacher(test_x)[..., :10].argmax(-1) == test_y).float().mean(dim=1)
        acc_student = (student(test_x)[..., :10].argmax(-1) == test_y).float().mean(dim=1)
    logger.log_baseline("Teacher_Clean_Accuracy", acc_teacher.tolist())
    logger.log_baseline("Student_Clean_Accuracy", acc_student.tolist())

    quadrants = [
        ("Teacher", "Teacher", V_t, teacher),
        ("Teacher", "Student", V_t, student),
        ("Student", "Teacher", V_s, teacher),
        ("Student", "Student", V_s, student)
    ]

    # Collect a subset of data points to save for the dense Figure 3 scatter correlation plot
    # To ensure high statistical representation without JSON bloat, we record 500 images per digit class
    scatter_data = []
    
    # Helper to calculate prediction probability of target digit class
    def get_class_prob(logits, d):
        probs = t.softmax(logits[..., :10], dim=-1) # Shape: (M, B)
        return probs[..., d]

    # 5. Multi-Digit Sweeps Loop
    # Track which (tgt_name, d, eps) random noise baselines have been logged already.
    # The random baseline is a property of the TARGET model only, independent of the source.
    # This prevents logging the same target-fragility baseline 2x (once per quadrant sharing a target).
    logged_random_baselines = set()
    for src_name, tgt_name, vectors, target_model in quadrants:
        # Determine source models to generate adversarial inputs
        source_model = teacher if src_name == "Teacher" else student
        source_vectors = V_t if src_name == "Teacher" else V_s
        source_centroids = Cent_t if src_name == "Teacher" else Cent_s
        
        print(f"\n=======================================================")
        print(f"Sweeping transfer quadrant: V{src_name} -> T{tgt_name}")
        print(f"=======================================================")

        for d in range(10):
            print(f"Digit {d}: filtering test set class samples...")
            digit_mask = (test_y == d)
            x_digit = test_x[:, digit_mask, :, :, :]  # Shape: (M, B, 1, 28, 28)
            y_digit = test_y[digit_mask].unsqueeze(0).expand(N_MODELS, -1)  # (M, B)
            
            B = x_digit.shape[1]
            if B == 0:
                continue

            # Compute clean predictions and valid mask (intersection of correctness)
            with t.no_grad():
                clean_pred_src = source_model(x_digit)[..., :10].argmax(-1)
                clean_pred_tgt = target_model(x_digit)[..., :10].argmax(-1)
                valid_mask = (clean_pred_src == d) & (clean_pred_tgt == d) # Shape: (M, B)

            n_scatter = min(B, 500)
            x_scatter = x_digit[:, :n_scatter, :, :, :]
            y_scatter = y_digit[:, :n_scatter]
            valid_mask_s = valid_mask[:, :n_scatter]

            targets_to_run = [g for g in range(10) if g != d]
            num_targets = len(targets_to_run)
            targets_tensor = t.tensor(targets_to_run, device=DEVICE) # (9,)

            # Pre-expand for vectorized batched execution over 9 targets simultaneously
            x_digit_flat = x_digit.unsqueeze(1).expand(-1, num_targets, -1, -1, -1, -1).flatten(1, 2)
            y_targets_flat = targets_tensor.unsqueeze(0).unsqueeze(2).expand(N_MODELS, -1, B).flatten(1, 2)
            x_scatter_flat = x_scatter.unsqueeze(1).expand(-1, num_targets, -1, -1, -1, -1).flatten(1, 2)
            y_targets_scatter_flat = targets_tensor.unsqueeze(0).unsqueeze(2).expand(N_MODELS, -1, n_scatter).flatten(1, 2)

            # ----------------------------------------------------
            # ATTACK 1: Input-Space PGD Epsilon Sweep
            # ----------------------------------------------------
            for eps in EPSILONS:
                x_adv_flat = pgd_attack_1(source_model, x_digit_flat, y_targets_flat, eps)
                x_adv = x_adv_flat.view(N_MODELS, num_targets, B, 1, 28, 28)
                
                with t.no_grad():
                    logits_adv_flat = target_model(x_adv_flat)
                    logits_adv = logits_adv_flat.view(N_MODELS, num_targets, B, TOTAL_OUT)
                    pred_adv = logits_adv[..., :10].argmax(-1) # (M, 9, B)
                    
                    target_digits = targets_tensor.view(1, num_targets, 1) # (1, 9, 1)
                    is_tsr = (pred_adv == target_digits)
                    is_usr = (pred_adv != d)
                    
                    mask_expanded = valid_mask.unsqueeze(1) # (M, 1, B)
                    mask_sum = mask_expanded.float().sum(dim=-1).clamp(min=1e-8) # Prevent division by zero
                    
                    tsr_mean = (is_tsr & mask_expanded).float().sum(dim=-1) / mask_sum # (M, 9)
                    usr_mean = (is_usr & mask_expanded).float().sum(dim=-1) / mask_sum # (M, 9)

                    act_clean = get_latent_activations(target_model, x_digit) # (M, B, 256)
                    act_clean_expanded = act_clean.unsqueeze(1).expand(-1, num_targets, -1, -1)
                    act_adv_flat = get_latent_activations(target_model, x_adv_flat)
                    act_adv = act_adv_flat.view(N_MODELS, num_targets, B, 256)
                    
                    latent_shift_all = t.norm(act_adv - act_clean_expanded, p=2, dim=-1) # (M, 9, B)
                    latent_shift = (latent_shift_all * mask_expanded.float()).sum(dim=-1) / mask_sum # (M, 9)

                tsr_overall = tsr_mean.mean(dim=1) # (M,)
                usr_overall = usr_mean.mean(dim=1) # (M,)
                latent_shift_overall = latent_shift.mean(dim=1) # (M,)

                logger.log_point(series_id=f"Attack1_TSR_V{src_name}_T{tgt_name}_Epsilon", group=f"Digit_{d}", x_label="epsilon", x_value=eps, raw_accuracies=tsr_overall.tolist(), target_model=tgt_name)
                logger.log_point(series_id=f"Attack1_USR_V{src_name}_T{tgt_name}_Epsilon", group=f"Digit_{d}", x_label="epsilon", x_value=eps, raw_accuracies=usr_overall.tolist(), target_model=tgt_name)
                logger.log_point(series_id=f"Attack1_Latent_Shift_V{src_name}_T{tgt_name}_Epsilon", group=f"Digit_{d}", x_label="epsilon", x_value=eps, raw_accuracies=latent_shift_overall.tolist(), target_model=tgt_name)

                for t_idx, j in enumerate(targets_to_run):
                    logger.log_point(series_id=f"Attack1_TSR_Confusion_V{src_name}_T{tgt_name}_Epsilon_{eps}", group=f"Inject_{d}", x_label="target_digit", x_value=j, raw_accuracies=tsr_mean[:, t_idx].tolist(), target_model=tgt_name)

                # Collect Scatter Data points at heatmap snapshot epsilon
                if abs(eps - HEATMAP_EPS) < 1e-5:
                    x_adv_s_flat = pgd_attack_1(source_model, x_scatter_flat, y_targets_scatter_flat, eps)
                    with t.no_grad():
                        logits_adv_s_flat = target_model(x_adv_s_flat)
                        logits_adv_s = logits_adv_s_flat.view(N_MODELS, num_targets, n_scatter, TOTAL_OUT)
                        
                        act_adv_s_flat = get_latent_activations(target_model, x_adv_s_flat)
                        act_adv_s = act_adv_s_flat.view(N_MODELS, num_targets, n_scatter, 256)
                        act_cl_s = get_latent_activations(target_model, x_scatter)
                        act_cl_s_expanded = act_cl_s.unsqueeze(1).expand(-1, num_targets, -1, -1)
                        
                        l_shift_s = t.norm(act_adv_s - act_cl_s_expanded, p=2, dim=-1) # (M, 9, n_scatter)
                        mask_s_expanded = valid_mask_s.unsqueeze(1).expand(-1, num_targets, -1) # (M, 9, n_scatter)
                        
                        p_clean_s = get_class_prob(target_model(x_scatter), d) # (M, n_scatter)
                        
                        for idx in range(n_scatter):
                            for t_idx, target_digit in enumerate(targets_to_run):
                                m_valid = mask_s_expanded[:, t_idx, idx]
                                if m_valid.sum() == 0: continue
                                p_adv = t.softmax(logits_adv_s[:, t_idx, idx, :10], dim=-1)[:, d]
                                conf_drop = p_clean_s[:, idx] - p_adv
                                mean_l_shift = l_shift_s[m_valid, t_idx, idx].mean().item()
                                mean_conf_drop = conf_drop[m_valid].mean().item()
                                scatter_data.append({"quadrant": f"V{src_name}_T{tgt_name}", "attack_type": 1, "latent_metric": float(mean_l_shift), "confidence_drop": float(mean_conf_drop), "src_digit": d, "target_digit": target_digit})

            # ----------------------------------------------------
            # ATTACK 2: Latent Representation Matching — Epsilon Sweep
            # Target: the target class activation centroid (source model's space)
            # Loss:   MSE( A_m(x_adv), mu_{target, m} )
            # ----------------------------------------------------
            with t.no_grad():
                # Target acts = target class centroid (shape: (N_MODELS, 256))
                # Expand to (N_MODELS, num_targets, B, 256)
                target_centroids_src = source_centroids[:, targets_to_run, :]  # (M, 9, 256)
                target_acts_base = target_centroids_src.unsqueeze(2).expand(-1, -1, B, -1)  # (M, 9, B, 256)
                target_acts_flat = target_acts_base.flatten(1, 2)  # (M, 9*B, 256)

                target_centroids_tgt = Cent_t if tgt_name == "Teacher" else Cent_s
                target_acts_tgt_base = target_centroids_tgt[:, targets_to_run, :].unsqueeze(2).expand(-1, -1, B, -1)  # (M, 9, B, 256)

            for eps2 in EPSILONS:
                x_adv_flat = pgd_attack_2(source_model, x_digit_flat, target_acts_flat, eps2)
                x_adv = x_adv_flat.view(N_MODELS, num_targets, B, 1, 28, 28)

                with t.no_grad():
                    logits_adv_flat = target_model(x_adv_flat)
                    logits_adv = logits_adv_flat.view(N_MODELS, num_targets, B, TOTAL_OUT)
                    pred_adv = logits_adv[..., :10].argmax(-1)

                    is_tsr = (pred_adv == target_digits)
                    is_usr = (pred_adv != d)
                    mask_sum = mask_expanded.float().sum(dim=-1).clamp(min=1e-8)

                    tsr_mean = (is_tsr & mask_expanded).float().sum(dim=-1) / mask_sum
                    usr_mean = (is_usr & mask_expanded).float().sum(dim=-1) / mask_sum

                    # Measure distance from achieved activations to the target centroid in TARGET model space
                    act_adv_flat = get_latent_activations(target_model, x_adv_flat)
                    act_adv = act_adv_flat.view(N_MODELS, num_targets, B, 256)
                    latent_dist_all = t.norm(act_adv - target_acts_tgt_base, p=2, dim=-1)
                    latent_dist = (latent_dist_all * mask_expanded.float()).sum(dim=-1) / mask_sum

                logger.log_point(series_id=f"Attack2_TSR_V{src_name}_T{tgt_name}_Epsilon", group=f"Digit_{d}", x_label="epsilon", x_value=eps2, raw_accuracies=tsr_mean.mean(dim=1).tolist(), target_model=tgt_name)
                logger.log_point(series_id=f"Attack2_USR_V{src_name}_T{tgt_name}_Epsilon", group=f"Digit_{d}", x_label="epsilon", x_value=eps2, raw_accuracies=usr_mean.mean(dim=1).tolist(), target_model=tgt_name)
                logger.log_point(series_id=f"Attack2_Latent_Distance_V{src_name}_T{tgt_name}_Epsilon", group=f"Digit_{d}", x_label="epsilon", x_value=eps2, raw_accuracies=latent_dist.mean(dim=1).tolist(), target_model=tgt_name)

                for t_idx, j in enumerate(targets_to_run):
                    logger.log_point(series_id=f"Attack2_TSR_Confusion_V{src_name}_T{tgt_name}_Epsilon_{eps2}", group=f"Inject_{d}", x_label="target_digit", x_value=j, raw_accuracies=tsr_mean[:, t_idx].tolist(), target_model=tgt_name)

                # Collect scatter data at heatmap epsilon snapshot
                if abs(eps2 - HEATMAP_EPS) < 1e-5:
                    with t.no_grad():
                        target_acts_s_base = target_centroids_src[:, :, :].unsqueeze(2).expand(-1, -1, n_scatter, -1).flatten(1, 2)

                    x_adv_s_flat = pgd_attack_2(source_model, x_scatter_flat, target_acts_s_base, eps2)

                    with t.no_grad():
                        logits_adv_s_flat = target_model(x_adv_s_flat)
                        logits_adv_s = logits_adv_s_flat.view(N_MODELS, num_targets, n_scatter, TOTAL_OUT)

                        act_adv_s = get_latent_activations(target_model, x_adv_s_flat).view(N_MODELS, num_targets, n_scatter, 256)
                        target_acts_tgt_s = target_centroids_tgt[:, targets_to_run, :].unsqueeze(2).expand(-1, -1, n_scatter, -1)  # (M, 9, n_scatter, 256)

                        l_dist_s = t.norm(act_adv_s - target_acts_tgt_s, p=2, dim=-1)
                        p_clean_s = get_class_prob(target_model(x_scatter), d)
                        mask_s_expanded = valid_mask_s.unsqueeze(1).expand(-1, num_targets, -1)  # (M, 9, n_scatter)

                        for idx in range(n_scatter):
                            for t_idx, target_digit in enumerate(targets_to_run):
                                m_valid = mask_s_expanded[:, t_idx, idx]
                                if m_valid.sum() == 0: continue
                                p_adv = t.softmax(logits_adv_s[:, t_idx, idx, :10], dim=-1)[:, d]
                                conf_drop = p_clean_s[:, idx] - p_adv
                                mean_l_dist = l_dist_s[m_valid, t_idx, idx].mean().item()
                                mean_conf_drop = conf_drop[m_valid].mean().item()
                                scatter_data.append({"quadrant": f"V{src_name}_T{tgt_name}", "attack_type": 2, "latent_metric": float(mean_l_dist), "confidence_drop": float(mean_conf_drop), "src_digit": d, "target_digit": target_digit})

    # Save Unified JSON Logs
    logger.save(SCRIPT_NAME)

    # Save Secondary Scatter Data JSON File
    scatter_path = "/home/eran.b/takehome/outputs/latent_steering_scatter.json"
    with open(scatter_path, "w") as f:
        json.dump(scatter_data, f, indent=2)
    print(f"\u2705 Secondary scatter data saved to: {scatter_path}")
    print("\n\U0001f389 Completed all Latent Representation Matching attacks and sweeps successfully!")
