import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration & Paths ---
OUTPUTS_DIR = "/home/eran.b/takehome/outputs"
PLOTS_DIR = "/home/eran.b/takehome/plots_a"
GRAPHS_DIR = "/home/eran.b/takehome/graphs__std_a"
JSON_FILE = os.path.join(OUTPUTS_DIR, "dropout_15e_stage.json")
OUT_ACC_PDF = os.path.join(GRAPHS_DIR, "4a_dropout_accuracy_sweep.pdf")
OUT_ACC_PNG = os.path.join(PLOTS_DIR, "4a_dropout_accuracy_sweep.png")
OUT_SIM_PDF = os.path.join(GRAPHS_DIR, "4b_dropout_similarity_sweep.pdf")
OUT_SIM_PNG = os.path.join(PLOTS_DIR, "4b_dropout_similarity_sweep.png")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# --- Aesthetic Standard (Matching tools/generate_std_plots.py) ---
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
    'lines.linewidth': 2
})

COLOR_MAP = {
    "Student-Only": "#2ca02c", # Green
    "Teacher-Only": "#1f77b4", # Blue
    "Both": "#d62728",         # Red (Symmetric)
}

MARKER_MAP = {
    "Student-Only": "o",
    "Teacher-Only": "s",
    "Both": "^"
}

def extract_series(data, series_id):
    rows = []
    for point in data.get("data_series", []):
        if point["series_id"] == series_id:
            rows.append({
                "x": point["x_axis"]["value"],
                "group": point["group"],
                "mean": point["metrics"].get("accuracy_mean", 0),
                "std": point["metrics"].get("accuracy_std", 0)
            })
    return pd.DataFrame(rows)

def main():
    print(f"Loading {JSON_FILE}...")
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    
    # Extract Baselines
    base_acc = data["baselines"]["No_Reg_Student_MNIST"]["accuracy_mean"]
    base_acc_std = data["baselines"]["No_Reg_Student_MNIST"]["accuracy_std"]
    base_sim = data["baselines"]["No_Reg_Avg_Cosine_Sim"]["accuracy_mean"]
    base_sim_std = data["baselines"]["No_Reg_Avg_Cosine_Sim"]["accuracy_std"]
    
    # Extract Data
    df_acc = extract_series(data, "Dropout_Sweep")
    df_sim = extract_series(data, "Avg_Cosine_Similarity")
    
    # --- Plotting Panel A: Accuracy ---
    plt.figure(figsize=(6, 5))
    for group in ["Student-Only", "Teacher-Only", "Both"]:
        gdf = df_acc[df_acc["group"] == group].sort_values("x")
        if gdf.empty: continue
        xs = np.concatenate([[0.0], gdf["x"].values])
        means = np.concatenate([[base_acc], gdf["mean"].values])
        stds = np.concatenate([[base_acc_std], gdf["std"].values])
        color = COLOR_MAP.get(group)
        marker = MARKER_MAP.get(group)
        plt.plot(xs, means, marker=marker, label=group, color=color)
        plt.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
    
    plt.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Chance')
    plt.title("Dropout Robustness (Accuracy)")
    plt.xlabel("Dropout Probability (p)")
    plt.ylabel("Student MNIST Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ACC_PDF)
    plt.savefig(OUT_ACC_PNG)
    plt.close()
    print(f"Accuracy graphs saved: {OUT_ACC_PDF}, {OUT_ACC_PNG}")
    
    # --- Plotting Panel B: Activation Similarity ---
    plt.figure(figsize=(6, 5))
    for group in ["Student-Only", "Teacher-Only", "Both"]:
        gdf = df_sim[df_sim["group"] == group].sort_values("x")
        if gdf.empty: continue
        xs = np.concatenate([[0.0], gdf["x"].values])
        means = np.concatenate([[base_sim], gdf["mean"].values])
        stds = np.concatenate([[base_sim_std], gdf["std"].values])
        color = COLOR_MAP.get(group)
        marker = MARKER_MAP.get(group)
        plt.plot(xs, means, marker=marker, label=group, color=color)
        plt.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
    
    plt.title("Representational Alignment")
    plt.xlabel("Dropout Probability (p)")
    plt.ylabel("S ↔ T Cosine Similarity")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_SIM_PDF)
    plt.savefig(OUT_SIM_PNG)
    plt.close()
    print(f"Similarity graphs saved: {OUT_SIM_PDF}, {OUT_SIM_PNG}")

    # --- Generate Markdown Table Data ---
    print("\n--- TABLE DATA ---")
    regimes = ["Student-Only", "Teacher-Only", "Both"]
    for regime in regimes:
        print(f"\n### {regime}")
        # Print baseline
        print(f"| **0.0 (Baseline)** | **{base_acc:.3f} \u00b1 {base_acc_std:.3f}** | **{base_sim:.3f} \u00b1 {base_sim_std:.3f}** | - | - |")
        
        # Get metrics for this regime
        r_acc = df_acc[df_acc["group"] == regime].sort_values("x")
        r_sim = df_sim[df_sim["group"] == regime].sort_values("x")
        
        # Need S vs Init and T vs Init (these don't have baselines in the same way in the JSON)
        df_si = extract_series(data, "Student_vs_Init_Cosine_Sim")
        df_ti = extract_series(data, "Teacher_vs_Init_Cosine_Sim")
        r_si = df_si[df_si["group"] == regime].sort_values("x")
        r_ti = df_ti[df_ti["group"] == regime].sort_values("x")
        
        for p in [0.1, 0.3, 0.5]:
            acc_val = r_acc[r_acc["x"] == p]
            sim_val = r_sim[r_sim["x"] == p]
            si_val = r_si[r_si["x"] == p]
            ti_val = r_ti[r_ti["x"] == p]
            
            acc_str = f"{acc_val['mean'].values[0]:.3f} \u00b1 {acc_val['std'].values[0]:.3f}" if not acc_val.empty else "-"
            sim_str = f"{sim_val['mean'].values[0]:.3f} \u00b1 {sim_val['std'].values[0]:.3f}" if not sim_val.empty else "-"
            si_str = f"{si_val['mean'].values[0]:.3f} \u00b1 {si_val['std'].values[0]:.3f}" if not si_val.empty else "-"
            ti_str = f"{ti_val['mean'].values[0]:.3f} \u00b1 {ti_val['std'].values[0]:.3f}" if not ti_val.empty else "-"
            
            print(f"| **{p}** | **{acc_str}** | **{sim_str}** | {si_str} | {ti_str} |")

if __name__ == "__main__":
    main()
