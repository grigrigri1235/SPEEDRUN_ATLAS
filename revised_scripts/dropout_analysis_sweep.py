"""
revised_scripts/dropout_analysis_sweep.py

Dropout Analysis Sweep — 15 Epochs
"""

import math
import os
import sys
import numpy as np
import torch as t
from torch import nn
from torchvision import datasets, transforms
from typing import Sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import UniLogger

# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
N_MODELS = 10
M_GHOST = 3
BATCH_SIZE = 1024
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 15
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias   = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

def mlp(n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)

class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], p_dropout: float = 0.0):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes, p_dropout)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def multilinear_layers(self):
        return [m for m in self.net if isinstance(m, MultiLinear)]

# ───────────────────────────── data helpers ──────────────────────────────────
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

# ─────────────────────────── train / distill ─────────────────────────────────
def ce_first10(logits, labels):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())

def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()

def distill(student, teacher, idx, src_x, epochs: int,
            gsnr_imgs=None, ghost_idx=None):
    teacher.eval(); student.train()
    opt = t.optim.Adam(student.parameters(), lr=LR)
    gsnr_per_epoch = {}

    # Epoch 0: GSNR before any training
    if gsnr_imgs is not None:
        gsnr_per_epoch[0] = get_gradient_gsnr(student, teacher, gsnr_imgs, ghost_idx)

    for ep in range(epochs):
        student.train()  # re-enable train mode after GSNR measurement
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean"
            )
            opt.zero_grad(); loss.backward(); opt.step()

        # GSNR after this epoch
        if gsnr_imgs is not None:
            gsnr_per_epoch[ep + 1] = get_gradient_gsnr(student, teacher, gsnr_imgs, ghost_idx)

    return gsnr_per_epoch

# ─────────────────────────────── metrics ─────────────────────────────────────
@t.inference_mode()
def get_accuracy(model, x, y):
    model.eval()
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()

@t.inference_mode()
def _cosine_per_model(vecs_a, vecs_b):
    return nn.functional.cosine_similarity(vecs_a, vecs_b, dim=1).tolist()

@t.inference_mode()
def get_activations(model, x):
    model.eval()
    curr = x.flatten(2)
    acts = []
    for layer in model.net:
        curr = layer(curr)
        if isinstance(layer, MultiLinear):
            acts.append(curr)
    return acts

@t.inference_mode()
def get_activation_similarities(model_a, model_b, ref_x):
    acts_a = get_activations(model_a, ref_x)
    acts_b = get_activations(model_b, ref_x)
    layer_sims = []
    all_flat_a, all_flat_b = [], []
    for aa, ab in zip(acts_a, acts_b):
        fa = aa.flatten(1)
        fb = ab.flatten(1)
        layer_sims.append(_cosine_per_model(fa, fb))
        all_flat_a.append(fa)
        all_flat_b.append(fb)
    full_a = t.cat(all_flat_a, dim=1)
    full_b = t.cat(all_flat_b, dim=1)
    avg_sim = _cosine_per_model(full_a, full_b)
    return avg_sim, layer_sims

@t.inference_mode()
def get_weight_stats(model):
    layers = model.multilinear_layers()
    stats = {}
    for i, layer in enumerate(layers):
        if i < len(layers) - 1:
            flat = t.cat([layer.weight.flatten(1), layer.bias], dim=1)
            stats[f"Layer{i}_Mag"] = flat.abs().mean(1).tolist()
        else:
            w_mnist = layer.weight[:, :10, :].flatten(1, 2)
            b_mnist = layer.bias[:, :10]
            m_mnist = t.cat([w_mnist, b_mnist], dim=1).abs().mean(1).tolist()
            w_ghost = layer.weight[:, 10:, :].flatten(1, 2)
            b_ghost = layer.bias[:, 10:]
            m_ghost = t.cat([w_ghost, b_ghost], dim=1).abs().mean(1).tolist()
            stats["Layer2_MNIST_Mag"] = m_mnist
            stats["Layer2_Ghost_Mag"] = m_ghost
    return stats

@t.inference_mode()
def get_gradient_gsnr(model, teacher, rand_imgs, ghost_idx):
    """
    Computes Gradient GSNR = ||E[grad]||^2 / sum(Var(grad)) intra-run.
    We isolate the ghost channel weights in Layer 2.
    """
    # CRITICAL: Model must be in train mode to activate dropout!
    # Teacher must be in eval mode as it is during distillation.
    model.train()
    teacher.eval()
    
    # Get hidden activations before final layer (Layer 2)
    flat_imgs = rand_imgs.flatten(2)
    hidden_s = model.net[:-1](flat_imgs) # (N_MODELS, B, 256)
    
    # Get logits
    logits_s = model.net[-1](hidden_s)   # (N_MODELS, B, 13)
    logits_t = teacher(rand_imgs)        # (N_MODELS, B, 13)
    
    # Ghost channel distillation loss (KL Divergence)
    # The gradient of KL(softmax(T) || softmax(S)) w.r.t S logits is softmax(S) - softmax(T)
    out = logits_s[:, :, ghost_idx]
    tgt = logits_t[:, :, ghost_idx]
    
    prob_s = nn.functional.softmax(out, dim=-1)
    prob_t = nn.functional.softmax(tgt, dim=-1)
    
    grad_logits = prob_s - prob_t # (N_MODELS, B, 3)
    
    # Per-sample gradient for W: grad_logits (B, 3, 1) * hidden_s (B, 1, 256) -> (B, 3, 256)
    per_sample_grads_w = t.einsum("mbo,mbi->mboi", grad_logits, hidden_s) # (10, B, 3, 256)
    per_sample_grads_b = grad_logits # (10, B, 3)
    
    # Flatten across the ghost weight/bias parameters
    flat_grads = t.cat([per_sample_grads_w.flatten(2), per_sample_grads_b], dim=2)
    
    # E[grad] over batch B
    mean_grad = flat_grads.mean(dim=1) # (N_MODELS, params)
    # Var(grad) over batch B
    var_grad  = flat_grads.var(dim=1)  # (N_MODELS, params)
    
    # GSNR per model = ||E[grad]||^2 / sum(Var(grad))
    signal = (mean_grad**2).sum(dim=1)
    noise  = var_grad.sum(dim=1) + 1e-12
    gsnr_per_model = (signal / noise).tolist()
    
    return gsnr_per_model

@t.inference_mode()
def get_weight_change_variance(model, reference):
    m_layers = model.multilinear_layers()
    r_layers = reference.multilinear_layers()
    stats = {}
    
    # Only look at the final classification layer (Layer 2)
    layer = m_layers[-1]
    ref_layer = r_layers[-1]
    
    dw = layer.weight - ref_layer.weight
    db = layer.bias - ref_layer.bias
    
    # MNIST channel (first 10)
    dw_mnist = dw[:, :10, :].flatten(1, 2)
    db_mnist = db[:, :10]
    flat_mnist = t.cat([dw_mnist, db_mnist], dim=1)
    var_mnist = flat_mnist.var(dim=0).tolist()
    mean_mnist = flat_mnist.mean(dim=0).tolist()
    
    # Ghost channel (last 3)
    dw_ghost = dw[:, 10:, :].flatten(1, 2)
    db_ghost = db[:, 10:]
    flat_ghost = t.cat([dw_ghost, db_ghost], dim=1)
    var_ghost = flat_ghost.var(dim=0).tolist()
    mean_ghost = flat_ghost.mean(dim=0).tolist()
    
    stats["Layer2_Weight_Change_Var_MNIST"] = var_mnist
    stats["Layer2_Weight_Change_Var_Ghost"] = var_ghost
    stats["Layer2_Weight_Change_Mean_MNIST"] = mean_mnist
    stats["Layer2_Weight_Change_Mean_Ghost"] = mean_ghost
    return stats

# ───────────────────────────── experiment runner ─────────────────────────────
def run_experiment(regime: str, intensity: float, train_x, train_y, test_x, test_y, rand_imgs, ref_x):
    p_t, p_s = 0.0, 0.0
    if regime == 'Student-Only': p_s = intensity
    elif regime == 'Teacher-Only': p_t = intensity
    elif regime == 'Both': p_t, p_s = intensity, intensity

    sizes = [28 * 28, 256, 256, TOTAL_OUT]
    reference = MultiClassifier(N_MODELS, sizes, p_dropout=0.0).to(DEVICE)

    teacher = MultiClassifier(N_MODELS, sizes, p_dropout=p_t).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)

    student = MultiClassifier(N_MODELS, sizes, p_dropout=p_s).to(DEVICE)
    student.load_state_dict(reference.state_dict())

    gsnr_per_epoch = distill(student, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL,
                             gsnr_imgs=rand_imgs[:, :512], ghost_idx=GHOST_IDX)

    avg_sim, layer_sims = get_activation_similarities(student, teacher, ref_x)
    student_vs_init, _ = get_activation_similarities(student, reference, ref_x)
    teacher_vs_init, _ = get_activation_similarities(teacher, reference, ref_x)

    t_stats = get_weight_stats(teacher)
    s_stats = get_weight_stats(student)
    
    t_var = get_weight_change_variance(teacher, reference)
    s_var = get_weight_change_variance(student, reference)

    teacher.eval()
    with t.no_grad():
        ghost_logits = teacher(rand_imgs[:, :512])[:, :, GHOST_IDX]
        t_ghost_logit_mag = ghost_logits.abs().mean(dim=(1, 2)).tolist()

    return {
        'student_mnist':     get_accuracy(student, test_x, test_y),
        'teacher_mnist':     get_accuracy(teacher, test_x, test_y),
        'avg_cosine_sim':    avg_sim,
        'layer_sims':        layer_sims,
        'student_vs_init':   student_vs_init,
        'teacher_vs_init':   teacher_vs_init,
        't_stats':           t_stats,
        's_stats':           s_stats,
        't_var':             t_var,
        's_var':             s_var,
        't_ghost_logit_mag': t_ghost_logit_mag,
        'gsnr_per_epoch':    gsnr_per_epoch,
        'gsnr':              gsnr_per_epoch.get(0, []),  # backward compat: init GSNR
    }

# ────────────────────────────── logging helper ───────────────────────────────
def log_all(logger, regime, lam, res):
    logger.log_point("Dropout_Sweep",               regime, "lambda", lam, res['student_mnist'],    target_model="Student")
    logger.log_point("Teacher_MNIST_Accuracy",      regime, "lambda", lam, res['teacher_mnist'],    target_model="Teacher")
    logger.log_point("Avg_Cosine_Similarity",       regime, "lambda", lam, res['avg_cosine_sim'],   target_model="Both")
    logger.log_point("Teacher_Ghost_Logit_Mag",     regime, "lambda", lam, res['t_ghost_logit_mag'], target_model="Teacher")
    logger.log_point("Student_vs_Init_Cosine_Sim",  regime, "lambda", lam, res['student_vs_init'],  target_model="Student")
    logger.log_point("Teacher_vs_Init_Cosine_Sim",  regime, "lambda", lam, res['teacher_vs_init'],  target_model="Teacher")
    logger.log_point("Ghost_GSNR",                  regime, "lambda", lam, res['gsnr'],             target_model="Student")

    for ep, gsnr_vals in res['gsnr_per_epoch'].items():
        # Metric across lambdas at a fixed epoch
        logger.log_point(f"Ghost_GSNR_Ep{ep}", regime, "lambda", lam, gsnr_vals, target_model="Student")
        # Metric across epochs at a fixed lambda (clean trajectory plotting)
        logger.log_point("Ghost_GSNR_Trajectory", f"{regime}_p{lam}", "epoch", ep, gsnr_vals, target_model="Student")

    for li in range(len(res['layer_sims'])):
        logger.log_point(f"Layer{li}_Cosine_Sim", regime, "lambda", lam, res['layer_sims'][li], target_model="Both")

    for prefix, stats in [("Teacher", res['t_stats']), ("Student", res['s_stats'])]:
        for key, vals in stats.items():
            logger.log_point(f"{prefix}_{key}", regime, "lambda", lam, vals, target_model=prefix)

    for prefix, var_stats in [("Teacher", res['t_var']), ("Student", res['s_var'])]:
        for key, vals in var_stats.items():
            logger.log_point(f"{prefix}_{key}", regime, "lambda", lam, vals, target_model=prefix)

# ────────────────────────────────── plotting ─────────────────────────────────
def plot_combined(regime_data: dict, lambdas: list, out_path: str):
    n_regimes = len(regime_data)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 4), sharey=True)
    if n_regimes == 1: axes = [axes]

    for ax, (regime, data) in zip(axes, regime_data.items()):
        xs    = np.array([d['lambda'] for d in data])
        stud  = np.array([d['student_mnist_mean']  for d in data])
        teach = np.array([d['teacher_mnist_mean']  for d in data])
        cosim = np.array([d['avg_cosine_sim_mean'] for d in data])
        stud_s  = np.array([d['student_mnist_std']  for d in data])
        teach_s = np.array([d['teacher_mnist_std']  for d in data])
        cosim_s = np.array([d['avg_cosine_sim_std'] for d in data])

        ax.plot(xs, stud,  'g-o',  label='Student MNIST Acc')
        ax.fill_between(xs, stud-stud_s,   stud+stud_s,   alpha=0.15, color='g')
        ax.plot(xs, teach, 'b-s',  label='Teacher MNIST Acc')
        ax.fill_between(xs, teach-teach_s, teach+teach_s, alpha=0.15, color='b')
        ax.plot(xs, cosim, 'r--^', label='Avg Cosine Sim')
        ax.fill_between(xs, cosim-cosim_s, cosim+cosim_s, alpha=0.15, color='r')
        ax.set_xlabel('Dropout Probability (p)')
        ax.set_title(f'Regime: {regime}')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Metric Value')
    fig.suptitle('Dropout Analysis 15E: Accuracy & Activation Divergence', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Combined plot saved: {out_path}")

# ─────────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    t.manual_seed(SEED)
    np.random.seed(SEED)
    train_ds, test_ds = get_mnist()
    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s,  test_y  = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x  = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    rand_imgs = t.rand_like(train_x) * 2 - 1

    SIM_BATCH = 1024
    ref_x = test_x[:, :SIM_BATCH].contiguous()

    logger = UniLogger(
        experiment_id="dropout_analysis_15e",
        target_model="Both",
        experiment_phase="Both",
        n_models=N_MODELS
    )

    probs = [0.1, 0.3, 0.5]
    regimes = ['Student-Only', 'Teacher-Only', 'Both']
    regime_plot_data = {r: [] for r in regimes}

    print("Running: Baseline (No Reg)...")
    base = run_experiment('Student-Only', 0.0, train_x, train_y, test_x, test_y, rand_imgs, ref_x)
    logger.log_baseline("No_Reg_Student_MNIST",  base['student_mnist'])
    logger.log_baseline("No_Reg_Teacher_MNIST",  base['teacher_mnist'])
    logger.log_baseline("No_Reg_Avg_Cosine_Sim", base['avg_cosine_sim'])
    logger.log_baseline("No_Reg_Student_vs_Init_Cosine_Sim", base['student_vs_init'])
    logger.log_baseline("No_Reg_Teacher_vs_Init_Cosine_Sim", base['teacher_vs_init'])
    
    n_layers = len(base['layer_sims'])
    for li in range(n_layers):
        logger.log_baseline(f"No_Reg_Layer{li}_Cosine_Sim", base['layer_sims'][li])

    for regime in regimes:
        for p in probs:
            print(f"Running: {regime} p={p}...")
            res = run_experiment(regime, p, train_x, train_y, test_x, test_y, rand_imgs, ref_x)
            log_all(logger, regime, p, res)

            pt = {
                'lambda':                   p,
                'student_mnist_mean':        float(np.mean(res['student_mnist'])),
                'student_mnist_std':         float(np.std(res['student_mnist'])),
                'teacher_mnist_mean':        float(np.mean(res['teacher_mnist'])),
                'teacher_mnist_std':         float(np.std(res['teacher_mnist'])),
                'avg_cosine_sim_mean':       float(np.mean(res['avg_cosine_sim'])),
                'avg_cosine_sim_std':        float(np.std(res['avg_cosine_sim'])),
                'student_vs_init_mean':      float(np.mean(res['student_vs_init'])),
                'student_vs_init_std':       float(np.std(res['student_vs_init'])),
                'teacher_vs_init_mean':      float(np.mean(res['teacher_vs_init'])),
                'teacher_vs_init_std':       float(np.std(res['teacher_vs_init'])),
                'ghost_logit_mag_mean':      float(np.mean(res['t_ghost_logit_mag'])),
                'ghost_logit_mag_std':       float(np.std(res['t_ghost_logit_mag'])),
                'layer2_mnist_mag_mean':     float(np.mean(res['t_stats']['Layer2_MNIST_Mag'])),
                'layer2_mnist_mag_std':      float(np.std(res['t_stats']['Layer2_MNIST_Mag'])),
                'layer2_ghost_mag_mean':     float(np.mean(res['t_stats']['Layer2_Ghost_Mag'])),
                'layer2_ghost_mag_std':      float(np.std(res['t_stats']['Layer2_Ghost_Mag'])),
                's_var_ghost_mean':          float(np.mean(res['s_var']['Layer2_Weight_Change_Var_Ghost'])),
                's_var_ghost_std':           float(np.std(res['s_var']['Layer2_Weight_Change_Var_Ghost'])),
                't_var_ghost_mean':          float(np.mean(res['t_var']['Layer2_Weight_Change_Var_Ghost'])),
                't_var_ghost_std':           float(np.std(res['t_var']['Layer2_Weight_Change_Var_Ghost'])),
                's_mean_mnist_mean':         float(np.mean(res['s_var']['Layer2_Weight_Change_Mean_MNIST'])),
                's_mean_mnist_std':          float(np.std(res['s_var']['Layer2_Weight_Change_Mean_MNIST'])),
                's_mean_ghost_mean':         float(np.mean(res['s_var']['Layer2_Weight_Change_Mean_Ghost'])),
                's_mean_ghost_std':          float(np.std(res['s_var']['Layer2_Weight_Change_Mean_Ghost'])),
                't_mean_mnist_mean':         float(np.mean(res['t_var']['Layer2_Weight_Change_Mean_MNIST'])),
                't_mean_mnist_std':          float(np.std(res['t_var']['Layer2_Weight_Change_Mean_MNIST'])),
                't_mean_ghost_mean':         float(np.mean(res['t_var']['Layer2_Weight_Change_Mean_Ghost'])),
                't_mean_ghost_std':          float(np.std(res['t_var']['Layer2_Weight_Change_Mean_Ghost'])),
                'gsnr_mean':                 float(np.mean(res['gsnr'])),
                'gsnr_std':                  float(np.std(res['gsnr'])),
                'gsnr_final_mean':           float(np.mean(res['gsnr_per_epoch'].get(EPOCHS_DISTILL, []))),
                'gsnr_final_std':            float(np.std(res['gsnr_per_epoch'].get(EPOCHS_DISTILL, []))),
            }
            for li in range(n_layers):
                pt[f'layer{li}_cosine_sim_mean'] = float(np.mean(res['layer_sims'][li]))
                pt[f'layer{li}_cosine_sim_std']  = float(np.std(res['layer_sims'][li]))
            regime_plot_data[regime].append(pt)

    logger.save("dropout_15e_stage.json")
    plot_combined(regime_plot_data, probs, "plots_a/dropout_analysis_15e_combined.pdf")
    print("✅ Finished dropout_analysis_15e.")
