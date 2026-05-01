"""
06_centering_mechanics.py
Mechanistic Investigation: Why does Student-Only centering boost Ghost transfer by ~27%?

Backbone: scripts/05_centering_sweep.py (untouched).
This script extends the original with:
  - 15-epoch distillation (temporal dynamics)
  - Hook position sweep (Layer 1 vs Layer 3)
  - Per-epoch logging of:
      * Ghost_Accuracy
      * Student_Grad_Bias / Student_Grad_Weights (Hypothesis 1: Gradient Dominance)
      * Layer1/Layer3_Activation_Sim (Geometric Alignment)
      * Variance_Explained_PC1 (Hypothesis 2: Spectral Masking)
  - All metrics logged with mean + std across N=10 ensemble via UniLogger.
"""
import math
import os
import sys
from typing import Sequence
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/eran.b/takehome")
from utils.logger import UniLogger
import torch as t
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
EPOCHS_DISTILL = 15 if not DEBUG else 2
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

# Architecture layout for sizes [784, 256, 256, 13]:
#   net[0] = MultiLinear(784 -> 256)
#   net[1] = ReLU                       <-- Hidden Layer 1 (HOOK TARGET "L1")
#   net[2] = MultiLinear(256 -> 256)
#   net[3] = ReLU                       <-- Hidden Layer 2 (HOOK TARGET "L3")
#   net[4] = MultiLinear(256 -> 13)
HOOK_POSITIONS = {
    "L1": 1,  # After first ReLU
    "L3": 3,  # After second ReLU (original target)
}

# Reference batch size for activation similarity
REF_BATCH_SIZE = 1024


# ─── Model Architecture (identical to 05_centering_sweep.py) ───

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


# ─── Data Loading (identical to 05_centering_sweep.py) ───

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


# ─── Training (identical to 05_centering_sweep.py) ───

def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train_teacher(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE, shuffle=True):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


# ─── Centering Hook ───

def centering_hook(module, input, output):
    """Batch mean centering: strip absolute coordinate geography, keep relative structure."""
    return output - output.mean(dim=1, keepdim=True)


# ─── Measurement Functions ───

@t.inference_mode()
def get_accuracy(model, x, y):
    """Per-model MNIST accuracy. Returns list of N_MODELS floats."""
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


@t.inference_mode()
def get_layer_activations(model, x, layer_idx):
    """
    Extract activations at a specific nn.Sequential layer index.
    Returns tensor of shape [N_MODELS, batch, hidden_dim].
    """
    # Register a temporary hook to capture the output
    activations = {}
    def capture_hook(module, inp, out):
        activations['out'] = out
    handle = model.net[layer_idx].register_forward_hook(capture_hook)
    _ = model(x)
    handle.remove()
    return activations['out']


@t.inference_mode()
def compute_activation_cosine_sim(model_a, model_b, ref_x, layer_idx):
    """
    Compute per-model cosine similarity between two models' activations
    at a given layer. Returns list of N_MODELS floats.
    """
    act_a = get_layer_activations(model_a, ref_x, layer_idx)  # [M, B, D]
    act_b = get_layer_activations(model_b, ref_x, layer_idx)  # [M, B, D]
    # Flatten batch dimension for each model: [M, B*D]
    flat_a = act_a.flatten(1)
    flat_b = act_b.flatten(1)
    cos = nn.functional.cosine_similarity(flat_a, flat_b, dim=1)  # [M]
    return cos.tolist()


@t.inference_mode()
def compute_variance_explained_pc1(model, ref_x, layer_idx):
    """
    Compute the fraction of variance explained by the first principal component
    of the activation matrix at a given layer. Returns list of N_MODELS floats.
    High values mean the mean vector dominates (spectral masking).
    """
    act = get_layer_activations(model, ref_x, layer_idx)  # [M, B, D]
    results = []
    for m in range(act.shape[0]):
        a = act[m]  # [B, D]
        # Center before PCA
        a_centered = a - a.mean(dim=0, keepdim=True)
        # Compute covariance eigenvalues via SVD (more stable than eig)
        _, s, _ = t.linalg.svd(a_centered, full_matrices=False)
        variances = s ** 2
        total_var = variances.sum()
        if total_var > 0:
            results.append((variances[0] / total_var).item())
        else:
            results.append(0.0)
    return results


def compute_grad_norms(student, layer_idx):
    """
    Compute gradient norms for bias vs weights of the final linear layer (net[4]).
    This measures how much gradient is spent on spatial translation (bias)
    vs feature learning (weights). Returns (bias_norms, weight_norms) as lists.
    """
    # Final linear layer is always net[4]
    final_layer = student.net[4]
    bias_norms = []
    weight_norms = []
    for m in range(N_MODELS):
        if final_layer.bias.grad is not None:
            bias_norms.append(final_layer.bias.grad[m].norm().item())
        else:
            bias_norms.append(0.0)
        if final_layer.weight.grad is not None:
            weight_norms.append(final_layer.weight.grad[m].norm().item())
        else:
            weight_norms.append(0.0)
    return bias_norms, weight_norms


# ─── Main Experiment Loop ───

def copy_matching_weights(src_model, dst_model):
    dst_state = dst_model.state_dict()
    for k, v in src_model.state_dict().items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k].copy_(v)
    dst_model.load_state_dict(dst_state)


def run_experiment(regime, hook_pos_name, hook_pos_idx, 
                   train_x, train_y, test_x, test_y, rand_imgs, ref_x, logger):
    """
    Run a single centering experiment for a given regime and hook position.
    Logs all metrics per epoch to the logger.
    """
    t.manual_seed(SEED)
    np.random.seed(SEED)

    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    regime_names = {'A': 'Standard', 'B': 'Student-Only', 'C': 'Teacher-Only', 'D': 'Both'}
    regime_name = regime_names[regime]

    # --- Setup models with shared init ---
    reference = MultiClassifier(N_MODELS, sizes).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    copy_matching_weights(reference, teacher)
    train_teacher(teacher, train_x, train_y, EPOCHS_TEACHER)

    student = MultiClassifier(N_MODELS, sizes).to(DEVICE)
    copy_matching_weights(reference, student)

    # --- Apply centering hooks ---
    handles = []
    if regime in ('B', 'D'):   # Student gets centering
        handles.append(student.net[hook_pos_idx].register_forward_hook(centering_hook))
    if regime in ('C', 'D'):   # Teacher gets centering
        handles.append(teacher.net[hook_pos_idx].register_forward_hook(centering_hook))

    # --- Distillation with per-epoch measurement ---
    teacher.eval()
    student.train()
    opt = t.optim.Adam(student.parameters(), lr=LR)

    for epoch in range(1, EPOCHS_DISTILL + 1):
        # --- Train one epoch ---
        student.train()
        for (bx,) in PreloadedDataLoader(rand_imgs, None, BATCH_SIZE, shuffle=True):
            with t.no_grad():
                tgt = teacher(bx)[:, :, GHOST_IDX]
            out = student(bx)[:, :, GHOST_IDX]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- Measure after this epoch ---

        # 1. Ghost Accuracy (need to temporarily handle eval hooks)
        for h in handles: h.remove()
        handles.clear()
        eval_handles = []
        if regime in ('B', 'D'):
            eval_handles.append(student.net[hook_pos_idx].register_forward_hook(centering_hook))
        
        accs = get_accuracy(student, test_x, test_y)
        
        for h in eval_handles: h.remove()
        eval_handles.clear()

        # 2. Gradient norms (from last backward pass of this epoch)
        bias_norms, weight_norms = compute_grad_norms(student, hook_pos_idx)

        # 3. Activation similarity at both layers (always measure both)
        # Remove centering hooks for raw similarity measurement
        sim_l1 = compute_activation_cosine_sim(student, teacher, ref_x, HOOK_POSITIONS["L1"])
        sim_l3 = compute_activation_cosine_sim(student, teacher, ref_x, HOOK_POSITIONS["L3"])

        # 4. Variance explained by PC1 (on the student's hooked layer)
        var_pc1 = compute_variance_explained_pc1(student, ref_x, hook_pos_idx)

        # --- Log all metrics ---
        suffix = hook_pos_name  # "L1" or "L3"

        logger.log_point(
            series_id=f"Ghost_Accuracy_{suffix}",
            group=regime_name, x_label="epoch", x_value=epoch,
            raw_accuracies=accs, target_model="Student"
        )
        logger.log_point(
            series_id=f"Student_Grad_Bias_{suffix}",
            group=regime_name, x_label="epoch", x_value=epoch,
            raw_accuracies=bias_norms, target_model="Student"
        )
        logger.log_point(
            series_id=f"Student_Grad_Weights_{suffix}",
            group=regime_name, x_label="epoch", x_value=epoch,
            raw_accuracies=weight_norms, target_model="Student"
        )
        logger.log_point(
            series_id=f"Layer1_Activation_Sim_{suffix}",
            group=regime_name, x_label="epoch", x_value=epoch,
            raw_accuracies=sim_l1, target_model="Student"
        )
        logger.log_point(
            series_id=f"Layer3_Activation_Sim_{suffix}",
            group=regime_name, x_label="epoch", x_value=epoch,
            raw_accuracies=sim_l3, target_model="Student"
        )
        logger.log_point(
            series_id=f"Variance_Explained_PC1_{suffix}",
            group=regime_name, x_label="epoch", x_value=epoch,
            raw_accuracies=var_pc1, target_model="Student"
        )

        print(f"  Epoch {epoch:2d} | Acc={np.mean(accs):.3f} | "
              f"GradBias={np.mean(bias_norms):.4f} | GradW={np.mean(weight_norms):.4f} | "
              f"SimL1={np.mean(sim_l1):.3f} SimL3={np.mean(sim_l3):.3f} | "
              f"PC1={np.mean(var_pc1):.3f}")

        # --- Reinstall centering hooks for next epoch ---
        if regime in ('B', 'D'):
            handles.append(student.net[hook_pos_idx].register_forward_hook(centering_hook))
        if regime in ('C', 'D'):
            handles.append(teacher.net[hook_pos_idx].register_forward_hook(centering_hook))

    # Cleanup
    for h in handles: h.remove()


# ─── Main ───

if __name__ == "__main__":
    print(f"🚀 06_centering_mechanics | Device: {DEVICE} | N={N_MODELS} | "
          f"Teacher={EPOCHS_TEACHER}ep | Distill={EPOCHS_DISTILL}ep")

    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    # Memory optimization: share noise across models
    rand_imgs = (t.rand_like(train_x_s) * 2 - 1).unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    # Reference batch for activation similarity (first 1024 test images)
    ref_x = test_x[:, :REF_BATCH_SIZE]

    logger = UniLogger(
        experiment_id="06_centering_mechanics",
        target_model="Multiple",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )

    regimes = {'A': 'Standard', 'B': 'Student-Only', 'C': 'Teacher-Only', 'D': 'Both'}

    for hook_name, hook_idx in HOOK_POSITIONS.items():
        for regime, regime_name in regimes.items():
            print(f"\n{'='*60}")
            print(f"Hook={hook_name} (net[{hook_idx}]) | Regime={regime_name}")
            print(f"{'='*60}")
            run_experiment(
                regime=regime,
                hook_pos_name=hook_name,
                hook_pos_idx=hook_idx,
                train_x=train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
                rand_imgs=rand_imgs,
                ref_x=ref_x,
                logger=logger
            )

    logger.save("centering_sweep_results.json")
    print("\n✅ Finished 06_centering_mechanics → outputs/centering_sweep_results.json")
