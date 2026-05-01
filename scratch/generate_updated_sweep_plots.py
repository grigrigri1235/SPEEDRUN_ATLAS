import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTPUTS_DIR = "outputs"
PLOTS_DIR = "graphs__std_a"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Aesthetic standard
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

def extract_series(data, series_id):
    rows = []
    for point in data.get("data_series", []):
        if point["series_id"] == series_id:
            rows.append({
                "x": point["x_axis"]["value"],
                "group": point["group"],
                "acc": point["metrics"].get("accuracy_mean", 0),
                "std": point["metrics"].get("accuracy_std", 0)
            })
    return pd.DataFrame(rows)

def plot_sweep(json_path, out_name, title, x_label):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get baseline
    baseline_acc = 0.519
    for b in data.get("baselines", []): # wait, in v5 it's a dict or list?
        # Let's check the head again... it was a dict!
        pass
    
    # Actually I'll just use the 0.519 I found earlier.
    if isinstance(data.get("baselines"), dict):
        baseline_acc = data["baselines"].get("No_Reg_Student_MNIST", {}).get("accuracy_mean", 0.519)
    
    df = extract_series(data, "Student_MNIST_Accuracy")
    
    plt.figure(figsize=(6, 4))
    for group in ["Student-Only", "Teacher-Only", "Both"]:
        gdf = df[df['group'] == group].sort_values('x')
        if len(gdf) == 0: continue
        
        label = "Symmetric" if group == "Both" else group
        color = COLOR_MAP.get(group)
        
        p = plt.plot(gdf['x'], gdf['acc'], marker='o', label=label, color=color)
        plt.fill_between(gdf['x'], gdf['acc'] - gdf['std'], gdf['acc'] + gdf['std'], color=color, alpha=0.15)

    plt.axhline(0.1, color='black', linestyle='--', alpha=0.4, label='Chance (0.1)')
    plt.axhline(baseline_acc, color='blue', linestyle=':', alpha=0.5, label=f'Baseline ({baseline_acc:.3f})')

    plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel("Transfer Accuracy")
    plt.title(title)
    plt.legend(fontsize=9, loc='upper right')
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    
    pdf_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated {pdf_path}")

if __name__ == "__main__":
    plot_sweep("outputs/l1_analysis_v5_results.json", 
               "2_l1_regularization_sweep.pdf", 
               "L1 Regularization Dynamics (Updated)", 
               "L1 Penalty (λ)")
    
    plot_sweep("outputs/l2_analysis_v2_results.json", 
               "3_l2_weight_decay_sweep.pdf", 
               "L2 Weight Decay Dynamics (Updated)", 
               "L2 Weight Decay (λ)")
