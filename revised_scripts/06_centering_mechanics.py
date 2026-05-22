"""
06_centering_mechanics.py
Mechanistic Investigation: Why does Student-Only centering boost Ghost transfer by ~27%?

Backbone: scripts/05_centering_sweep.py (untouched).
This script extends the original with:
  - 10-epoch distillation (temporal dynamics)
  - Hook position sweep (Layer 1 vs Layer 3)
  - THREE experimental conditions:
      1. Baseline:  ReLU student, no centering (Regime A)
      2. Tanh arm:  Tanh student, no centering (Regime A) — does zero-mean geometry help?
      3. Centering: ReLU student, Student-Only centering hook (Regime B) — hypothesis: bias grows
  - Per-epoch logging of:
      * Ghost_Accuracy       (MNIST accuracy on real test images — the subliminal transfer signal)
      * Student_Bias_WeightNorm  (actual |b| parameter — expected to shoot up in centering arm)
      * Student_Grad_Bias / Student_Grad_Weights (gradient norms)
      * Layer1/Layer3_Activation_Sim (Geometric Alignment)
      * Variance_Explained_PC1 (Spectral Masking)
  - All metrics logged with mean + std across N=10 ensemble via UniLogger.
"""
import math
import os
import sys
from typing import Sequence, Type
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
EPOCHS_DISTILL = 10 if not DEBUG else 2
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

# Architecture layout for sizes [784, 256, 256, 13]:
#   net[0] = MultiLinear(784 -> 256)
#   net[1] = ActFn                       <-- Hidden Layer 1 (HOOK TARGET "L1")
#   net[2] = MultiLinear(256 -> 256)
#   net[3] = ActFn                       <-- Hidden Layer 2 (HOOK TARGET "L3")
#   net[4] = MultiLinear(256 -> 13)
HOOK_POSITIONS = {
    "L1": 1,  # After first activation
    "L3": 3,  # After second activation (original target)
}

# Reference batch size for activation similarity
REF_BATCH_SIZE = 1024


# ─── Model Architecture (identical to 05_centering_sweep.py, + act_fn param) ───

class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]


def mlp(n_models: int, sizes: Sequence[int], act_fn: Type[nn.Module] = nn.ReLU):
    """act_fn defaults to ReLU, matching 05_centering_sweep.py behaviour."""
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(act_fn())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], act_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes, act_fn)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))


# ─── Data Loading (identical to 05_centering_sweep.py) ───

def get_mnist():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root = "~/.pytorch/MNIST_data/"
    return (datasets.MNIST(root, download=True, train=True, transform=tfm),
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


# ─── Centering Hook (identical to 05_centering_sweep.py) ───

def centering_hook(module, input, output):
    """Batch mean centering: strip absolute coordinate geography, keep relative structure."""
    return output - output.mean(dim=1, keepdim=True)


# ─── Shared Init (identical to 05_centering_sweep.py) ───

def copy_matching_weights(src_model, dst_model):
    """Copy weights from src to dst where shapes match."""
    dst_state = dst_model.state_dict()
    for k, v in src_model.state_dict().items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k].copy_(v)
    dst_model.load_state_dict(dst_state)


# ─── Measurement Functions ───

@t.inference_mode()
def get_accuracy(model, x, y):
    """
    Per-model MNIST accuracy on real test images vs ground truth labels.
    This is the PRIMARY metric — it measures subliminal transfer of MNIST structure.
    Matches 05_centering_sweep.py exactly.
    """
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


@t.inference_mode()
def get_layer_activations(model, x, layer_idx):
    """Extract activations at a specific nn.Sequential layer index. [M, B, D]."""
    activations = {}
    def capture_hook(module, inp, out):
        activations['out'] = out
    handle = model.net[layer_idx].register_forward_hook(capture_hook)
    _ = model(x)
    handle.remove()
    return activations['out']


@t.inference_mode()
def compute_activation_cosine_sim(model_a, model_b, ref_x, layer_idx):
    """Per-model cosine similarity between two models' activations at a layer."""
    act_a = get_layer_activations(model_a, ref_x, layer_idx)
    act_b = get_layer_activations(model_b, ref_x, layer_idx)
    flat_a = act_a.flatten(1)   # [M, B*D]
    flat_b = act_b.flatten(1)
    return nn.functional.cosine_similarity(flat_a, flat_b, dim=1).tolist()


@t.inference_mode()
def compute_variance_explained_pc1(model, ref_x, layer_idx):
    """
    Fraction of variance explained by the first principal component.
    High values indicate the mean vector dominates (spectral masking).
    """
    act = get_layer_activations(model, ref_x, layer_idx)
    results = []
    for m in range(act.shape[0]):
        a = act[m]
        a_centered = a - a.mean(dim=0, keepdim=True)
        _, s, _ = t.linalg.svd(a_centered, full_matrices=False)
        variances = s ** 2
        total_var = variances.sum()
        results.append((variances[0] / total_var).item() if total_var > 0 else 0.0)
    return results


def compute_grad_norms(student):
    """Gradient norms for bias vs weights of final linear layer (net[4])."""
    final_layer = student.net[4]
    bias_norms, weight_norms = [], []
    for m in range(N_MODELS):
        bias_norms.append(
            final_layer.bias.grad[m].norm().item() if final_layer.bias.grad is not None else 0.0)
        weight_norms.append(
            final_layer.weight.grad[m].norm().item() if final_layer.weight.grad is not None else 0.0)
    return bias_norms, weight_norms


@t.inference_mode()
def compute_bias_weight_norms(model):
    """
    Actual bias PARAMETER norms (not gradients) of the final linear layer.
    The centering hypothesis predicts these grow large when centering is applied,
    because the model must learn a large b to compensate for mean-subtraction.
    Returns list of N_MODELS floats.
    """
    final_layer = model.net[4]
    return [final_layer.bias[m].norm().item() for m in range(N_MODELS)]


# ─── Main Experiment Loop ───

def run_experiment(regime: str, hook_pos_name: str, hook_pos_idx: int,
                   train_x, train_y, test_x, test_y, rand_imgs, ref_x, logger,
                   student_act_fn: Type[nn.Module] = nn.ReLU):
    """
    Run a single centering experiment.

    regime:         'A' = Standard, 'B' = Student-Only centering
    hook_pos_name:  'L1' or 'L3'
    hook_pos_idx:   1 or 3
    student_act_fn: nn.ReLU (baseline) or nn.Tanh (Tanh arm)
    Teacher is always ReLU.
    """
    t.manual_seed(SEED)
    np.random.seed(SEED)

    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    regime_names = {'A': 'Standard', 'B': 'Student-Only', 'C': 'Teacher-Only', 'D': 'Both'}
    act_name = "Tanh" if student_act_fn == nn.Tanh else "ReLU"
    full_group_name = f"{act_name}_{regime_names[regime]}"

    # --- Shared Init (identical to 05_centering_sweep.py) ---
    reference = MultiClassifier(N_MODELS, sizes).to(DEVICE)   # always ReLU

    teacher = MultiClassifier(N_MODELS, sizes).to(DEVICE)      # always ReLU
    copy_matching_weights(reference, teacher)
    train_teacher(teacher, train_x, train_y, EPOCHS_TEACHER)

    student = MultiClassifier(N_MODELS, sizes, act_fn=student_act_fn).to(DEVICE)
    copy_matching_weights(reference, student)  # same initial weights, different activation

    # --- Apply centering hooks (identical logic to 05_centering_sweep.py) ---
    handles = []
    if regime in ('B', 'D'):   # Student gets centering
        handles.append(student.net[hook_pos_idx].register_forward_hook(centering_hook))
    if regime in ('C', 'D'):   # Teacher gets centering
        handles.append(teacher.net[hook_pos_idx].register_forward_hook(centering_hook))

    # --- Distillation with per-epoch measurement ---
    teacher.eval()
    opt = t.optim.Adam(student.parameters(), lr=LR)

    for epoch in range(1, EPOCHS_DISTILL + 1):
        # --- Train one epoch on GHOST_IDX only, from noise (identical to 05) ---
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

        # Temporarily remove training hooks for clean evaluation
        for h in handles: h.remove()
        handles.clear()

        # Reinstate centering for eval if student was trained with it (matches 05)
        eval_handles = []
        if regime in ('B', 'D'):
            eval_handles.append(
                student.net[hook_pos_idx].register_forward_hook(centering_hook))

        # PRIMARY METRIC: MNIST accuracy on real test images (subliminal transfer)
        accs = get_accuracy(student, test_x, test_y)

        for h in eval_handles: h.remove()
        eval_handles.clear()

        # DIAGNOSTIC METRICS
        bias_grad_norms, weight_grad_norms = compute_grad_norms(student)
        bias_weight_norms = compute_bias_weight_norms(student)
        sim_l1 = compute_activation_cosine_sim(student, teacher, ref_x, HOOK_POSITIONS["L1"])
        sim_l3 = compute_activation_cosine_sim(student, teacher, ref_x, HOOK_POSITIONS["L3"])
        var_pc1 = compute_variance_explained_pc1(student, ref_x, hook_pos_idx)

        # GCS: Gradient Cosine Similarity (the advisor's hypothesis)
        # Measures alignment between MNIST-head and Ghost-head gradients at net[2].weight.
        # Two separate backward passes on a ref batch; centering hooks active (training state).
        student.train()
        with t.no_grad():
            tgt_ref = teacher(ref_x)          # [M, B, 13]
        out_ref = student(ref_x)              # [M, B, 13]  — differentiable
        opt.zero_grad()
        l_mnist_ref = nn.functional.kl_div(
            nn.functional.log_softmax(out_ref[:, :, list(range(10))], -1),
            nn.functional.softmax(tgt_ref[:, :, list(range(10))].detach(), -1),
            reduction="batchmean"
        )
        l_mnist_ref.backward(retain_graph=True)
        g_mnist = student.net[2].weight.grad.clone().detach().flatten(1)  # [M, D*D]
        opt.zero_grad()
        l_ghost_ref = nn.functional.kl_div(
            nn.functional.log_softmax(out_ref[:, :, GHOST_IDX], -1),
            nn.functional.softmax(tgt_ref[:, :, GHOST_IDX].detach(), -1),
            reduction="batchmean"
        )
        l_ghost_ref.backward()
        g_ghost = student.net[2].weight.grad.clone().detach().flatten(1)  # [M, D*D]
        gcs = nn.functional.cosine_similarity(g_mnist, g_ghost, dim=1).tolist()  # [M]
        opt.zero_grad()  # clean up

        # --- Log all metrics ---
        sfx = hook_pos_name
        logger.log_point(f"Ghost_Accuracy_{sfx}", full_group_name,
                         "epoch", epoch, accs, "Student")
        logger.log_point(f"Student_Bias_WeightNorm_{sfx}", full_group_name,
                         "epoch", epoch, bias_weight_norms, "Student")
        logger.log_point(f"Student_Grad_Bias_{sfx}", full_group_name,
                         "epoch", epoch, bias_grad_norms, "Student")
        logger.log_point(f"Student_Grad_Weights_{sfx}", full_group_name,
                         "epoch", epoch, weight_grad_norms, "Student")
        logger.log_point(f"Layer1_Activation_Sim_{sfx}", full_group_name,
                         "epoch", epoch, sim_l1, "Student")
        logger.log_point(f"Layer3_Activation_Sim_{sfx}", full_group_name,
                         "epoch", epoch, sim_l3, "Student")
        logger.log_point(f"Variance_Explained_PC1_{sfx}", full_group_name,
                         "epoch", epoch, var_pc1, "Student")
        logger.log_point(f"Gradient_Cosine_Similarity_{sfx}", full_group_name,
                         "epoch", epoch, gcs, "Student")

        print(f"  Ep {epoch:2d} | Acc={np.mean(accs):.3f} | GCS={np.mean(gcs):.4f} | "
              f"BiasNorm={np.mean(bias_weight_norms):.4f} | "
              f"GradBias={np.mean(bias_grad_norms):.4f} | "
              f"SimL1={np.mean(sim_l1):.3f} SimL3={np.mean(sim_l3):.3f} | "
              f"PC1={np.mean(var_pc1):.3f}")

        # Reinstall centering hooks for next epoch
        if regime in ('B', 'D'):
            handles.append(student.net[hook_pos_idx].register_forward_hook(centering_hook))
        if regime in ('C', 'D'):
            handles.append(teacher.net[hook_pos_idx].register_forward_hook(centering_hook))

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

    # Memory optimization: share noise across models (identical to 05)
    rand_imgs = (t.rand_like(train_x_s) * 2 - 1).unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    # Reference batch for activation similarity (first 1024 test images)
    ref_x = test_x[:, :REF_BATCH_SIZE]

    logger = UniLogger(
        experiment_id="06_centering_mechanics",
        target_model="Multiple",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )

    # Three experimental conditions:
    # 1. Baseline:  ReLU student, no centering
    # 2. Tanh arm:  Tanh student, no centering — does zero-mean geometry help?
    # 3. Centering: ReLU student, Student-Only centering — hypothesis: bias grows
    SWEEP = [
        (nn.ReLU, 'A'),   # Baseline
        (nn.Tanh, 'A'),   # Tanh variant
        (nn.ReLU, 'B'),   # Centered ReLU
    ]

    for hook_name, hook_idx in HOOK_POSITIONS.items():
        for student_act, regime in SWEEP:
            act_label = "Tanh" if student_act == nn.Tanh else "ReLU"
            print(f"\n{'='*60}")
            print(f"Hook={hook_name} (net[{hook_idx}]) | Student={act_label} | Regime={regime}")
            print(f"{'='*60}")
            run_experiment(
                regime=regime,
                hook_pos_name=hook_name,
                hook_pos_idx=hook_idx,
                train_x=train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
                rand_imgs=rand_imgs,
                ref_x=ref_x,
                logger=logger,
                student_act_fn=student_act,
            )

    logger.save("centering_sweep_results.json")
    print("\n✅ Finished 06_centering_mechanics → outputs/centering_sweep_results.json")
