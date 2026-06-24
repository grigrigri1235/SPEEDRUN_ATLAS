import math
import os
import sys
import argparse
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

SCRIPT_NAME     = "08_boundary_attack"

# ──────────────────────────── Core Modules ───────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias   = nn.Parameter(t.zeros(n_models, d_out))
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

    def get_penultimate(self, x: t.Tensor):
        h = x.flatten(2)
        for i in range(len(self.net) - 1):
            h = self.net[i](h)
        return h

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

# ──────────────────────── Boundary Attack ────────────────────────────────────
def boundary_attack_batch(model, x_clean, x_target, max_iters=500):
    """
    Vectorized Decision Boundary Attack.
    model: MultiClassifier
    x_clean: [M, B, C, H, W] - origin images (Class A)
    x_target: [M, B, C, H, W] - starting images (Class B)
    """
    x_adv = x_target.clone()
    M, B = x_clean.shape[:2]
    
    delta = t.ones(M, B, 1, 1, 1, device=x_clean.device) * 0.05
    epsilon = t.ones(M, B, 1, 1, 1, device=x_clean.device) * 0.01
    
    with t.no_grad():
        y_target = model(x_adv)[..., :10].argmax(-1)
        
    for i in range(max_iters):
        # 1. Orthogonal step
        noise = t.randn_like(x_adv)
        direction = x_clean - x_adv
        direction_norm = t.norm(direction.flatten(2), dim=2, keepdim=True).view(M,B,1,1,1) + 1e-8
        direction_unit = direction / direction_norm
        
        noise_proj = t.sum(noise * direction_unit, dim=(2,3,4), keepdim=True)
        noise_orth = noise - noise_proj * direction_unit
        noise_orth_unit = noise_orth / (t.norm(noise_orth.flatten(2), dim=2, keepdim=True).view(M,B,1,1,1) + 1e-8)
        
        x_orth = x_adv + delta * direction_norm * noise_orth_unit
        x_orth = t.clamp(x_orth, -1.0, 1.0)
        
        # 2. Concentric step
        dir_orth = x_clean - x_orth
        dir_orth_norm = t.norm(dir_orth.flatten(2), dim=2, keepdim=True).view(M,B,1,1,1) + 1e-8
        dir_orth_unit = dir_orth / dir_orth_norm
        
        x_new = x_orth + epsilon * dir_orth_norm * dir_orth_unit
        x_new = t.clamp(x_new, -1.0, 1.0)
        
        with t.no_grad():
            preds = model(x_new)[..., :10].argmax(-1)
            
        success = (preds == y_target).float().view(M,B,1,1,1)
        
        x_adv = t.where(success > 0, x_new, x_adv)
        
        epsilon = t.where(success > 0, epsilon * 1.05, epsilon * 0.95)
        delta = t.where(success > 0, delta * 1.05, delta * 0.95)
        
        epsilon = t.clamp(epsilon, 1e-5, 0.5)
        delta = t.clamp(delta, 1e-5, 0.5)
        
    return x_adv

# ─────────────────────────────── Sweeps ──────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run a smaller pilot sweep")
    args = parser.parse_args()

    n_samples_per_pair = 10 if args.pilot else 100
    suffix = "pilot" if args.pilot else "full"
    
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

    # 1. Train Teacher
    print("Training Teacher Ensemble...")
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    # 2. Distill Student
    print("Distilling Student Ensemble...")
    student = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    distill(student, teacher, rand_imgs, EPOCHS_DISTILL)

    logger = UniLogger(SCRIPT_NAME, "Both", "Transfer", N_MODELS)

    # Pre-select samples for each digit
    digit_indices_clean = {d: t.where(test_y == d)[0][:n_samples_per_pair * 9] for d in range(10)}
    digit_indices_target = {d: t.where(test_y == d)[0][:n_samples_per_pair * 9] for d in range(10)}

    print(f"Running Boundary Attacks ({suffix} mode, {n_samples_per_pair} per pair)...")
    
    # Pre-calculate boundary points
    adv_points = {"Teacher": {}, "Student": {}}
    
    for src_name, src_model in [("Teacher", teacher), ("Student", student)]:
        print(f"Generating boundary points on {src_name}...")
        adv_points[src_name] = {d: {} for d in range(10)}
        for src_d in range(10):
            for tgt_d in range(10):
                if src_d == tgt_d: continue
                idx_offset = (tgt_d if tgt_d < src_d else tgt_d - 1) * n_samples_per_pair
                idx_slice = slice(idx_offset, idx_offset + n_samples_per_pair)
                
                clean_idx = digit_indices_clean[src_d][idx_slice]
                target_idx = digit_indices_target[tgt_d][idx_slice]
                
                min_len = min(len(clean_idx), len(target_idx))
                clean_idx = clean_idx[:min_len]
                target_idx = target_idx[:min_len]
                
                x_clean_batch = test_x[:, clean_idx]
                x_target_batch = test_x[:, target_idx]
                
                x_adv_src = boundary_attack_batch(src_model, x_clean_batch, x_target_batch, max_iters=500)
                adv_points[src_name][src_d][tgt_d] = x_adv_src
                
                distances = t.norm((x_adv_src - x_clean_batch).flatten(2), dim=2) # [M, B]
                mean_dist = distances.mean(dim=1).cpu().tolist() # list of N_MODELS distances
                
                logger.log_point(
                    series_id=f"Boundary_Distance_V{src_name}",
                    x_label="target_digit", x_value=tgt_d,
                    group=f"Source_Digit_{src_d}",
                    raw_accuracies=mean_dist
                )

                # Calculate Latent Traversed Distance
                with t.no_grad():
                    z_clean = src_model.get_penultimate(x_clean_batch) # [M, B, 256]
                    z_adv = src_model.get_penultimate(x_adv_src) # [M, B, 256]
                latent_traversed_dists = t.norm(z_adv - z_clean, p=2, dim=2) # [M, B]
                mean_latent_traversed = latent_traversed_dists.mean(dim=1).cpu().tolist()

                logger.log_point(
                    series_id=f"Boundary_Latent_Distance_Traversed_V{src_name}",
                    x_label="target_digit", x_value=tgt_d,
                    group=f"Source_Digit_{src_d}",
                    raw_accuracies=mean_latent_traversed
                )

                # Calculate Latent Analytical Distance
                # Final linear head weights and biases
                W_s = src_model.net[-1].weight[:, src_d, :] # [M, 256]
                W_t = src_model.net[-1].weight[:, tgt_d, :] # [M, 256]
                b_s = src_model.net[-1].bias[:, src_d] # [M]
                b_t = src_model.net[-1].bias[:, tgt_d] # [M]

                diff_W = W_s - W_t # [M, 256]
                diff_b = b_s - b_t # [M]

                num = t.sum(diff_W.unsqueeze(1) * z_clean, dim=-1) + diff_b.unsqueeze(1) # [M, B]
                denom = t.norm(diff_W, p=2, dim=-1, keepdim=True) # [M, 1]
                latent_analytical_dists = t.abs(num) / (denom + 1e-8) # [M, B]
                mean_latent_analytical = latent_analytical_dists.mean(dim=1).cpu().tolist()

                logger.log_point(
                    series_id=f"Boundary_Latent_Distance_Analytical_V{src_name}",
                    x_label="target_digit", x_value=tgt_d,
                    group=f"Source_Digit_{src_d}",
                    raw_accuracies=mean_latent_analytical
                )

    # Now evaluate transfer
    for src_name, tgt_name, tgt_model in [
        ("Teacher", "Teacher", teacher),
        ("Student", "Student", student),
        ("Teacher", "Student", student),
        ("Student", "Teacher", teacher)
    ]:
        print(f"Evaluating Transfer: {src_name} -> {tgt_name}")
        for src_d in range(10):
            for tgt_d in range(10):
                if src_d == tgt_d: continue
                
                x_adv_src = adv_points[src_name][src_d][tgt_d]
                with t.no_grad():
                    preds_tgt = tgt_model(x_adv_src)[..., :10].argmax(-1)
                
                successes = (preds_tgt == tgt_d).float().mean(dim=1).cpu().tolist()
                
                logger.log_point(
                    series_id=f"Boundary_Transfer_V{src_name}_T{tgt_name}",
                    x_label="target_digit", x_value=tgt_d,
                    group=f"Source_Digit_{src_d}",
                    raw_accuracies=successes
                )

    logger.save(f"boundary_attack_{suffix}.json")
    print("Done!")
