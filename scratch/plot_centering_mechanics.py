import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = os.path.expanduser("~/takehome/outputs")
PLOTS_A_DIR = os.path.expanduser("~/takehome/plots_a")
GRAPHS_STD_DIR = os.path.expanduser("~/takehome/graphs__std_a")

os.makedirs(PLOTS_A_DIR, exist_ok=True)
os.makedirs(GRAPHS_STD_DIR, exist_ok=True)

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
    "Both": "#d62728",         # Red
    "Standard": "#7f7f7f",     # Gray
}

def load_data():
    path = os.path.join(OUTPUTS_DIR, "centering_sweep_results.json")
    with open(path, 'r') as f:
        return json.load(f)

def extract_series(data, series_id):
    rows = []
    if not data or "data_series" not in data:
        return pd.DataFrame()
    for point in data["data_series"]:
        if point["series_id"] == series_id:
            rows.append({
                "x": point["x_axis"]["value"],
                "group": point["group"],
                "mean": point["metrics"].get("accuracy_mean", 0),
                "std": point["metrics"].get("accuracy_std", 0)
            })
    return pd.DataFrame(rows)

def plot_line_clean(df, x_label, y_label, title, filename_base, log_y=False):
    plt.figure(figsize=(6, 4))
    
    if len(df) == 0:
        plt.close()
        return

    groups = df['group'].unique()
    for group in groups:
        gdf = df[df['group'] == group].sort_values('x')
        color = COLOR_MAP.get(group, None)
        p = plt.plot(gdf['x'], gdf['mean'], marker='o', label=group, color=color)
        if 'std' in gdf.columns:
            plt.fill_between(gdf['x'], gdf['mean'] - gdf['std'], gdf['mean'] + gdf['std'], color=p[0].get_color(), alpha=0.2)

    if log_y:
        plt.yscale('log')
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    png_path = os.path.join(PLOTS_A_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(GRAPHS_STD_DIR, f"{filename_base}.pdf")
    
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  [OK] Saved {filename_base} (.png, .pdf)")

def main():
    print("Generating Centering Mechanics Plots...")
    data = load_data()
    
    if not data:
        print("Data not found!")
        return

    # 1. Ghost Accuracy (L3 - Bottleneck)
    df_acc_l3 = extract_series(data, "Ghost_Accuracy_L3")
    plot_line_clean(df_acc_l3, "Epoch", "Ghost Transfer Accuracy", "Accuracy Trajectory (L3 Centering)", "centering_accuracy_trajectory_l3")

    # 2. Ghost Accuracy (L1 - Early)
    df_acc_l1 = extract_series(data, "Ghost_Accuracy_L1")
    plot_line_clean(df_acc_l1, "Epoch", "Ghost Transfer Accuracy", "Accuracy Trajectory (L1 Centering)", "centering_accuracy_trajectory_l1")

    # 3. Gradient Bias Norms (L3)
    df_grad_bias_l3 = extract_series(data, "Student_Grad_Bias_L3")
    plot_line_clean(df_grad_bias_l3, "Epoch", "Gradient Norm (Bias)", "Gradient Dominance: Final Layer Bias", "centering_grad_bias_l3")
    
    # 3b. Gradient Bias Norms (L3) Log Scale to see Both vs Standard clearly
    plot_line_clean(df_grad_bias_l3, "Epoch", "Gradient Norm (Bias) - Log Scale", "Gradient Dominance (Log Scale)", "centering_grad_bias_l3_log", log_y=True)

    # 4. Activation Similarity (L3)
    df_sim_l3 = extract_series(data, "Layer3_Activation_Sim_L3")
    plot_line_clean(df_sim_l3, "Epoch", "Activation Cosine Similarity", "Geometric Alignment (Layer 3)", "centering_activation_sim_l3")

    # 5. Spectral Masking / PC1 Variance (L3)
    df_pc1_l3 = extract_series(data, "Variance_Explained_PC1_L3")
    plot_line_clean(df_pc1_l3, "Epoch", "Fraction of Variance Explained", "Spectral Masking: PC1 Variance", "centering_pc1_variance_l3")

    print("[SUCCESS] All plots generated.")

if __name__ == "__main__":
    main()
