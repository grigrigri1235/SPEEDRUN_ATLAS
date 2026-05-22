"""
revised_scripts/generate_gsnr_plots.py

Generates premium GSNR phase transition plots for the NeurIPS report.
Overrides legacy dropout graphs with high-granularity (13-point) data.
Visualizes the "Static Hook" (Bias vs Weight GSNR resilience).
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ───────────────────────────────── settings ──────────────────────────────────
MASTER_JSON = "/home/eran.b/takehome/outputs/mechanism_sweep_results.json"
PDF_DIR = "/home/eran.b/takehome/graphs__std_a"
PNG_DIR = "/home/eran.b/takehome/plots_a"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# Aesthetic standard (Matching tools/generate_std_plots.py)
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5
})

COLOR_MAP = {
    "Student-Only": "#2ca02c", # Green
    "Teacher-Only": "#1f77b4", # Blue
    "Symmetric": "#d62728",    # Red (Both)
    "Weights": "#9467bd",      # Purple
    "Bias": "#ff7f0e",         # Orange
}

def format_group(g):
    if g == "Both": return "Symmetric"
    return g

# ─────────────────────────────── data loading ────────────────────────────────
def load_data():
    with open(MASTER_JSON, "r") as f:
        data = json.load(f)
    return data["data_series"]

def get_df(series_list, series_id):
    rows = []
    for s in series_list:
        if s["series_id"] == series_id:
            rows.append({
                "x": s["x_axis"]["value"],
                "group": format_group(s["group"]),
                "y": s["metrics"]["accuracy_mean"],
                "std": s["metrics"]["accuracy_std"]
            })
    return pd.DataFrame(rows).sort_values("x")

# ───────────────────────────────── plotting ──────────────────────────────────
def save_dual(name):
    plt.savefig(os.path.join(PDF_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(PNG_DIR, f"{name}.png"))
    print(f"  ✅ Saved {name} (PDF/PNG)")

def plot_accuracy(series_list):
    df = get_df(series_list, "Dropout_Sweep")
    plt.figure(figsize=(7, 5))
    for group in df["group"].unique():
        sub = df[df["group"] == group]
        p = plt.plot(sub["x"], sub["y"], marker='o', label=group, color=COLOR_MAP.get(group))
        plt.fill_between(sub["x"], sub["y"] - sub["std"], sub["y"] + sub["std"], alpha=0.15, color=p[0].get_color())
    
    plt.axhline(0.1, color='black', linestyle='--', alpha=0.5, label='Chance')
    plt.xlabel("Dropout Probability (p)")
    plt.ylabel("Transfer Accuracy")
    plt.title("The Ghost Wall: Phase Transition")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    save_dual("4a_dropout_accuracy_sweep")

def plot_similarity(series_list):
    df = get_df(series_list, "Avg_Cosine_Similarity")
    plt.figure(figsize=(7, 5))
    for group in df["group"].unique():
        sub = df[df["group"] == group]
        p = plt.plot(sub["x"], sub["y"], marker='s', label=group, color=COLOR_MAP.get(group))
        plt.fill_between(sub["x"], sub["y"] - sub["std"], sub["y"] + sub["std"], alpha=0.15, color=p[0].get_color())
    
    plt.xlabel("Dropout Probability (p)")
    plt.ylabel("Activation Cosine Similarity")
    plt.title("Representational Alignment Collapse")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    save_dual("4b_dropout_similarity_sweep")

def plot_gsnr_sweep(series_list):
    # Focus on Student-Only to show the decoupling clearly
    regime = "Student-Only"
    # We will compute the average across all epochs (0-15) for each p
    metrics = {
        "L3_Weights": "Total Avg Weights (L3)",
        "L3_Bias":    "Total Avg Bias (L3) - Static Hook",
        "L2_Weights": "Total Avg Penultimate Weights (L2)",
        "L2_Bias":    "Total Avg Penultimate Bias (L2)"
    }
    
    plt.figure(figsize=(8, 6))
    for sid, label in metrics.items():
        # Compute mean across all Ep0...Ep15
        p_vals = sorted(list(set([s["x_axis"]["value"] for s in series_list if s["x_axis"]["name"] == "lambda"])))
        y_means = []
        for p in p_vals:
            vals = []
            for ep in range(16):
                target_sid = f"Ghost_GSNR_{sid}_Ep{ep}"
                for s in series_list:
                    if s["series_id"] == target_sid and s["group"] == regime and s["x_axis"]["value"] == p:
                        vals.append(s["metrics"]["accuracy_mean"])
            y_means.append(np.mean(vals) if vals else 0.0)
            
        color = COLOR_MAP.get("Bias") if "Bias" in label else COLOR_MAP.get("Weights")
        style = '-' if "L3" in label else '--'
        plt.plot(p_vals, y_means, marker='o', label=label, color=color, linestyle=style)
    
    plt.axhline(0.0, color='red', linestyle='-', linewidth=1.5, alpha=0.8, label='Noise Floor (GSNR=0)')
    plt.yscale('linear')
    plt.xlabel("Dropout Probability (p)")
    plt.ylabel("Sustained GSNR (Avg Epochs 0-15)")
    plt.title("The Sustained Static Hook\n(Student-Only Regime)")
    plt.legend(fontsize=10)
    plt.ylim(-0.5, 10) 
    plt.tight_layout()
    save_dual("4c_dropout_gsnr_sweep")

def plot_gsnr_trajectory(series_list):
    # Show GSNR over time for p=0.5 (Collapse) vs p=0.0 (Baseline)
    plt.figure(figsize=(7, 5))
    
    # p=0.5 Student-Only L3 Weights vs Bias
    # Trajectory series IDs: Ghost_GSNR_L3_Weights_Trajectory
    # group: Student-Only_p0.5
    targets = [
        ("Student-Only_p0.0", "L3_Weights", "Baseline (p=0.0) Weights", "black", "-"),
        ("Student-Only_p0.5", "L3_Weights", "Collapse (p=0.5) Weights", COLOR_MAP["Weights"], "-"),
        ("Student-Only_p0.5", "L3_Bias",    "Collapse (p=0.5) Bias",    COLOR_MAP["Bias"], "-"),
    ]
    
    for group, mtype, label, color, style in targets:
        df = get_df(series_list, f"Ghost_GSNR_{mtype}_Trajectory")
        sub = df[df["group"] == group]
        if not sub.empty:
            p = plt.plot(sub["x"], sub["y"], marker='.', label=label, color=color, linestyle=style)
            plt.fill_between(sub["x"], sub["y"] - sub["std"], sub["y"] + sub["std"], alpha=0.1, color=p[0].get_color())

    plt.yscale('symlog', linthresh=0.1)
    # We remove ylim entirely to let symlog handle it, or adjust if needed.
    plt.xlabel("Epoch")
    plt.ylabel("GSNR (Log Scale)")
    plt.title("GSNR Evolution Across Distillation")
    plt.legend()
    plt.tight_layout()
    save_dual("4d_dropout_gsnr_trajectory")

def plot_gsnr_decline_all_regimes(series_list):
    # Show L3 Weights GSNR across all three regimes (Teacher-Only, Student-Only, Both)
    # Using Total Average across epochs
    regimes = ["Student-Only", "Teacher-Only", "Symmetric"]
    p_vals = sorted(list(set([s["x_axis"]["value"] for s in series_list if s["x_axis"]["name"] == "lambda"])))
    
    plt.figure(figsize=(7, 5))
    for r in regimes:
        group_name = "Both" if r == "Symmetric" else r
        y_means = []
        for p in p_vals:
            vals = []
            for ep in range(16):
                target_sid = f"Ghost_GSNR_L3_Weights_Ep{ep}"
                for s in series_list:
                    if s["series_id"] == target_sid and s["group"] == group_name and s["x_axis"]["value"] == p:
                        vals.append(s["metrics"]["accuracy_mean"])
            y_means.append(np.mean(vals) if vals else 0.0)
        
        plt.plot(p_vals, y_means, marker='o', label=r, color=COLOR_MAP.get(r))
    
    plt.axhline(0.0, color='red', linestyle='--', alpha=0.5, label='Noise Floor (GSNR=0)')
    plt.yscale('linear')
    plt.ylim(-0.5, 10)
    plt.xlabel("Dropout Probability (p)")
    plt.ylabel("Total Avg Weight GSNR")
    plt.title("GSNR Decline Across Noise Regimes")
    plt.legend()
    plt.tight_layout()
    save_dual("4f_gsnr_decline_all_regimes")

def plot_temporal_blackout(series_list):
    # Bar chart for p=0.5: Early, Mid, Late phases
    p_val = 0.5
    regime = "Student-Only"
    phases = [("Early (0-5)", 0, 5), ("Mid (6-10)", 6, 10), ("Late (11-15)", 11, 15)]
    
    w_avgs, b_avgs = [], []
    for label, start, end in phases:
        w_vals, b_vals = [], []
        for ep in range(start, end + 1):
            for s in series_list:
                if s["group"] == regime and s["x_axis"]["value"] == p_val:
                    if s["series_id"] == f"Ghost_GSNR_L3_Weights_Ep{ep}":
                        w_vals.append(s["metrics"]["accuracy_mean"])
                    elif s["series_id"] == f"Ghost_GSNR_L3_Bias_Ep{ep}":
                        b_vals.append(s["metrics"]["accuracy_mean"])
        w_avgs.append(np.mean(w_vals) if w_vals else 0.0)
        b_avgs.append(np.mean(b_vals) if b_vals else 0.0)

    x = np.arange(len(phases))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, w_avgs, width, label='Weights (L3)', color=COLOR_MAP["Weights"], alpha=0.8)
    ax.bar(x + width/2, b_avgs, width, label='Biases (L3)', color=COLOR_MAP["Bias"], alpha=0.8)
    
    ax.set_ylabel('Avg GSNR (at p=0.5)')
    ax.set_title('The Temporal Blackout: Weight vs Bias Survival')
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in phases])
    ax.legend()
    
    # Annotate the "Blackout"
    ax.annotate('Weight Blackout', xy=(1, w_avgs[1]), xytext=(1, w_avgs[1]+2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.tight_layout()
    save_dual("4g_gsnr_temporal_blackout")

def plot_resilience_bar(series_list):
    # Ratio at p=0.5
    p_val = 0.5
    regime = "Student-Only"
    
    w_df = get_df(series_list, "Ghost_GSNR_L3_Weights_Ep10")
    b_df = get_df(series_list, "Ghost_GSNR_L3_Bias_Ep10")
    
    w_val = w_df[(w_df["group"] == regime) & (w_df["x"] == p_val)]["y"].values[0]
    b_val = b_df[(b_df["group"] == regime) & (b_df["x"] == p_val)]["y"].values[0]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(["Weights", "Biases"], [w_val, b_val], color=[COLOR_MAP["Weights"], COLOR_MAP["Bias"]], alpha=0.8)
    
    # Annotate ratio
    ratio = b_val / w_val if w_val > 0 else float('inf')
    plt.text(0.5, max(w_val, b_val) * 0.7, f"Bias Resilience:\n{ratio:.1f}x Higher", 
             ha='center', fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.ylabel("GSNR at p=0.5 (Epoch 10)")
    plt.title("The Static Hook Resilience Factor (Ep 10)")
    plt.tight_layout()
    save_dual("4e_gsnr_bias_resilience")

# ─────────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Generating High-Granularity GSNR Plots...")
    series = load_data()
    
    plot_accuracy(series)
    plot_similarity(series)
    plot_gsnr_sweep(series)
    plot_gsnr_trajectory(series)
    plot_gsnr_decline_all_regimes(series)
    plot_temporal_blackout(series)
    plot_resilience_bar(series)
    
    print("\n✨ All GSNR plots generated and dual-exported.")
