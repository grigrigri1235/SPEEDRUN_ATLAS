"""
Visualization Script for Latent Steering & Adversarial Attacks
==============================================================
Generates high-fidelity, publication-grade figures from the outputs of
latent_steering_attacks.py (both the main JSON and the scatter JSON).
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Config & Aesthetics ---
JSON_PATH       = "outputs/latent_steering_attacks.json"
SCATTER_PATH    = "outputs/latent_steering_scatter.json"
PLOTS_DIR       = "plots_a"
os.makedirs(PLOTS_DIR, exist_ok=True)

# High-fidelity styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 300
})

QUADRANT_COLORS = {
    "VTeacher_TTeacher": "#2E86C1",  # Deep Blue (Control)
    "VTeacher_TStudent": "#E74C3C",  # Bright Red (Teacher -> Student)
    "VStudent_TTeacher": "#16A085",  # Teal (Student -> Teacher)
    "VStudent_TStudent": "#F39C12"   # Orange (Student -> Student)
}

QUADRANT_LABELS = {
    "VTeacher_TTeacher": "Teacher -> Teacher (Control)",
    "VTeacher_TStudent": "Teacher -> Student (Distill Transfer)",
    "VStudent_TTeacher": "Student -> Teacher (Reverse Transfer)",
    "VStudent_TStudent": "Student -> Student (Distill Consistency)"
}

def load_json_data():
    with open(JSON_PATH, 'r') as f:
        return json.load(f)

def load_scatter_data():
    with open(SCATTER_PATH, 'r') as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Robustness & Transferability Sweep Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_robustness_curves(data):
    print("Plotting Figure 1: Robustness & Transferability Curves...")
    
    epsilons = [0.05, 0.1, 0.2, 0.3]
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    quadrants = [
        ("Teacher", "Teacher", "VTeacher_TTeacher"),
        ("Teacher", "Student", "VTeacher_TStudent"),
        ("Student", "Teacher", "VStudent_TTeacher"),
        ("Student", "Student", "VStudent_TStudent")
    ]
    
    # ── Subplot A: PGD Epsilon Sweep ──
    ax_a = axes[0]
    for src, tgt, key in quadrants:
        success_means = []
        success_stds = []
        for eps in epsilons:
            sid = f"Attack1_Accuracy_V{src}_T{tgt}_Epsilon"
            points = [s for s in data["data_series"] if s["series_id"] == sid and s["x_axis"]["value"] == eps]
            
            # Remaining accuracy across all 10 digits
            accs = [p["metrics"]["accuracy_mean"] for p in points]
            # Fooling rate = 1 - Accuracy
            fooling_rates = [1.0 - acc for acc in accs]
            
            success_means.append(np.mean(fooling_rates))
            success_stds.append(np.std(fooling_rates))
            
        ax_a.plot(epsilons, success_means, label=QUADRANT_LABELS[key], marker='o', 
                  linewidth=3, color=QUADRANT_COLORS[key])
        ax_a.fill_between(epsilons, np.array(success_means) - np.array(success_stds), 
                           np.array(success_means) + np.array(success_stds), 
                           color=QUADRANT_COLORS[key], alpha=0.1)

    # Plot Target-Model Fragility Baselines under Random Noise (averaged over 5 seeds)
    for target, label, color in [
        ("Teacher", "Teacher Target Fragility (Random)", QUADRANT_COLORS["VTeacher_TTeacher"]),
        ("Student", "Student Target Fragility (Random)", QUADRANT_COLORS["VStudent_TStudent"])
    ]:
        success_means = []
        for eps in epsilons:
            sid = f"Random_Noise_Accuracy_T{target}_Epsilon"
            points = [s for s in data["data_series"] if s["series_id"] == sid and s["x_axis"]["value"] == eps]
            if not points:
                print(f"[WARN] No random noise baseline found for {sid} at eps={eps}. Skipping.")
                success_means.append(float('nan'))
                continue
            accs = [p["metrics"]["accuracy_mean"] for p in points]
            fooling_rates = [1.0 - acc for acc in accs]
            success_means.append(np.mean(fooling_rates))
            
        ax_a.plot(epsilons, success_means, label=label, linestyle=':', linewidth=2.5,
                  color=color, marker='x', markersize=6, alpha=0.8)

    # 90% Chance Line
    ax_a.axhline(y=0.90, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax_a.text(0.05, 0.92, "90% Chance Baseline (Random / Scrambled)", color='dimgray', fontsize=10, fontweight='bold')
    
    # Add Clean Baseline Tick (representing starting success rate ~ 1 - 0.98 = 2%)
    ax_a.plot(0.0, 0.02, marker='_', color='black', markersize=10, mew=3)
    ax_a.text(-0.015, 0.02, "Clean", ha='right', va='center', fontsize=10, fontweight='bold')
    
    ax_a.set_title("Input PGD Epsilon Sweep\n(Varying Input Budget ε)", pad=15)
    ax_a.set_xlabel("Perturbation Budget (Epsilon)")
    ax_a.set_ylabel("Attack Success Rate (1 - Accuracy)")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_xlim(-0.02, 0.32)
    ax_a.legend(loc='lower right', frameon=True, shadow=True)

    # ── Subplot B: Latent Steering Alpha Sweep ──
    ax_b = axes[1]
    for src, tgt, key in quadrants:
        success_means = []
        success_stds = []
        for alpha in alphas:
            sid = f"Attack2_FPR_V{src}_T{tgt}_Alpha"
            points = [s for s in data["data_series"] if s["series_id"] == sid and s["x_axis"]["value"] == alpha]
            
            # The logged value in accuracy_mean is the targeted FPR
            fprs = [p["metrics"]["accuracy_mean"] for p in points]
            
            success_means.append(np.mean(fprs))
            success_stds.append(np.std(fprs))
            
        ax_b.plot(alphas, success_means, label=QUADRANT_LABELS[key], marker='s', 
                  linewidth=3, color=QUADRANT_COLORS[key])
        ax_b.fill_between(alphas, np.array(success_means) - np.array(success_stds), 
                           np.array(success_means) + np.array(success_stds), 
                           color=QUADRANT_COLORS[key], alpha=0.1)

    # 90% Chance Line
    ax_b.axhline(y=0.90, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax_b.text(0.5, 0.92, "90% Chance Baseline", color='dimgray', fontsize=10, fontweight='bold')
    
    # Add Clean Baseline Tick
    ax_b.plot(0.0, 0.02, marker='_', color='black', markersize=10, mew=3)
    
    ax_b.set_title("Latent Steering Alpha Sweep\n(Varying Steering Dosage α at fixed ε = 0.1)", pad=15)
    ax_b.set_xlabel("Steering Intensity (Alpha)")
    ax_b.set_ylabel("Steering Success Rate (Targeted FPR)")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_xlim(-0.2, 5.2)
    ax_b.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.suptitle("Robustness & Transferability Sweep Curves", fontsize=22, y=0.98)
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/attack_sweep_curves.png", bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2a & 2b: Confusion Heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def get_confusion_matrix(data, src, tgt, key, param_type, param_val):
    # Retrieve confusion matrix data
    if param_type == "Epsilon":
        sid = f"Attack1_Confusion_V{src}_T{tgt}_Epsilon_{param_val}"
    else:
        sid = f"Attack2_Confusion_V{src}_T{tgt}_Alpha_{param_val}"
        
    matrix = np.zeros((10, 10))
    for s in data["data_series"]:
        if s["series_id"] == sid:
            inject = int(s["group"].split("_")[1])
            target_digit = int(s["x_axis"]["value"])
            matrix[inject, target_digit] = s["metrics"]["accuracy_mean"]
    return matrix

def plot_confusion_heatmaps(data, attack_num, param_type, param_val, filename):
    print(f"Plotting Figure 2{'a' if attack_num==1 else 'b'}: {filename}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    quadrants = [
        ("Teacher", "Teacher", "VTeacher_TTeacher", axes[0, 0]),
        ("Teacher", "Student", "VTeacher_TStudent", axes[0, 1]),
        ("Student", "Teacher", "VStudent_TTeacher", axes[1, 0]),
        ("Student", "Student", "VStudent_TStudent", axes[1, 1])
    ]
    
    for src, tgt, key, ax in quadrants:
        matrix = get_confusion_matrix(data, src, tgt, key, param_type, param_val)
        
        # Use warm colormap where higher numbers indicate larger transitions (hijacking)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True, ax=ax,
                    vmin=0.0, vmax=1.0, linewidths=0.5, linecolor="#DDDDDD",
                    annot_kws={"size": 9, "weight": "bold"})
        
        ax.set_title(QUADRANT_LABELS[key], fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Label (Outputs)", fontsize=11)
        ax.set_ylabel("Original Image Digit (True Class)", fontsize=11)
        ax.set_xticklabels(range(10))
        ax.set_yticklabels(range(10))
        
        # Draw borders around the diagonal to highlight survival rate
        for idx in range(10):
            ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='blue', lw=1.5, alpha=0.6))
            
    title_suffix = f"(Input PGD Sweep at ε = {param_val})" if attack_num == 1 else f"(Latent Steering Sweep at α = {param_val})"
    plt.suptitle(f"Multi-Digit Vulnerability Confusion Matrices\n{title_suffix}", fontsize=20, y=0.97)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    fig.savefig(f"{PLOTS_DIR}/{filename}", bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Internal Latent-Space Shift vs. Outer Adversarial Success
# ─────────────────────────────────────────────────────────────────────────────
def plot_latent_shift_correlations(scatter_data):
    print("Plotting Figure 3: Latent Shift Correlations...")
    
    df = pd.DataFrame(scatter_data)
    if df.empty:
        print("⚠️ No scatter data found. Skipping Figure 3.")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    attack_configs = [
        (1, axes[0], "Attack 1: Input PGD (ε = 0.1)\nLatent Shift vs. Class Confidence Drop", "Latent L2 Shift: ||A₂(x*) - A₂(x)|₂"),
        (2, axes[1], "Attack 2: Latent Steering (α = 1.0)\nDistance to Target vs. Class Confidence Drop", "Target L2 Distance: ||A₂(x*) - T(x, α)|₂")
    ]
    
    for att_type, ax, title, xlabel in attack_configs:
        sub_df = df[df["attack_type"] == att_type]
        
        for quad, color in QUADRANT_COLORS.items():
            quad_df = sub_df[sub_df["quadrant"] == quad]
            if quad_df.empty:
                continue
                
            x_vals = quad_df["latent_metric"].values
            y_vals = quad_df["confidence_drop"].values
            
            # Plot the dense scatter cloud with small dots and high transparency
            ax.scatter(x_vals, y_vals, color=color, s=4, alpha=0.2, label=None)
            
            # Add linear regression fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
            x_range = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            y_range = slope * x_range + intercept
            
            r_sq = r_value ** 2
            ax.plot(x_range, y_range, color=color, linewidth=2.5,
                    label=f"{QUADRANT_LABELS[quad]} (R² = {r_sq:.3f})")
            
        ax.set_title(title, pad=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("True Digit Class Probability Drop\n(P_clean - P_adv)")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='lower right', frameon=True, shadow=True, markerscale=2)

    plt.suptitle("Internal Representational Shift vs. Outer Adversarial Hijacking", fontsize=22, y=0.98)
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/latent_shift_correlations.png", bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading data files...")
    try:
        main_data = load_json_data()
        scat_data = load_scatter_data()
        
        # Figure 1
        plot_robustness_curves(main_data)
        
        # Figure 2a
        plot_confusion_heatmaps(main_data, attack_num=1, param_type="Epsilon", param_val=0.1, 
                                filename="attack1_confusion_heatmaps.png")
                                
        # Figure 2b
        plot_confusion_heatmaps(main_data, attack_num=2, param_type="Alpha", param_val=1.0, 
                                filename="attack2_confusion_heatmaps.png")
                                
        # Figure 3
        plot_latent_shift_correlations(scat_data)
        
        print(f"\n🎉 All high-fidelity visualization plots successfully saved in `{PLOTS_DIR}/`!")
    except Exception as e:
        print(f"❌ Error during visualization execution: {e}")
        import traceback
        traceback.print_exc()
