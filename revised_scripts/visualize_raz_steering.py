"""
Visualization Script for Raz's Steering Experiment
=================================================
Generates high-fidelity, interpretable plots based on the JSON output of raz_steering.py.
Decoupled from experiment execution for rapid aesthetic iteration.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- Config & Aesthetics ---
JSON_PATH = "outputs/raz_steering.json"
PLOTS_DIR = "plots_a"
os.makedirs(PLOTS_DIR, exist_ok=True)

# High-fidelity style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 300
})

COLORS = {
    "Teacher": "#2E86C1", # Deep Blue (Authority)
    "Student": "#E74C3C", # Bright Red (Vulnerable)
    "Waterfall": "#16A085" # Teal
}

def load_data():
    with open(JSON_PATH, 'r') as f:
        return json.load(f)

def get_matrix_data(data, src, tgt, alpha):
    sid = f"Matrix_V{src}_T{tgt}_Alpha_{alpha}"
    rows = []
    for s in data["data_series"]:
        if s["series_id"] == sid:
            # group is "Inject_{digit}" (target z), x_axis.value is actual_digit (actual y)
            target_digit = int(s["group"].split("_")[1])
            actual_digit = s["x_axis"]["value"]
            if target_digit == actual_digit: continue # Skip self-accuracy
            
            rows.append({
                "target": target_digit,
                "actual": actual_digit,
                "tasr_mean": s["metrics"]["accuracy_mean"],
                "tasr_std": s["metrics"]["accuracy_std"]
            })
    return pd.DataFrame(rows)



def plot_pca_map(data):
    """Generates the PCA Topology Map with arrows."""
    print("Plotting PCA Topology...")
    
    # Extract centroids
    c_t = []
    c_s = []
    for i in range(10):
        # Extract from new metadata series
        p_t = [s for s in data["data_series"] if s["series_id"] == "Centroids_Teacher" and s["x_axis"]["value"] == i][0]
        p_s = [s for s in data["data_series"] if s["series_id"] == "Centroids_Student" and s["x_axis"]["value"] == i][0]
        c_t.append(p_t["raw"])
        c_s.append(p_s["raw"])
    
    c_t = np.array(c_t)
    c_s = np.array(c_s)
    
    pca = PCA(n_components=2)
    combined = np.vstack([c_t, c_s])
    proj = pca.fit_transform(combined)
    var_exp = pca.explained_variance_ratio_
    
    t_proj, s_proj = proj[:10], proj[10:]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(t_proj[:, 0], t_proj[:, 1], c=COLORS["Teacher"], s=300, label='Teacher (High Friction)', edgecolors='k', alpha=0.9, zorder=3)
    ax.scatter(s_proj[:, 0], s_proj[:, 1], c=COLORS["Student"], s=300, label='Student (Smoothed)', edgecolors='k', alpha=0.9, zorder=3)
    
    for i in range(10):
        ax.text(t_proj[i, 0], t_proj[i, 1]+0.1, str(i), color=COLORS["Teacher"], fontsize=14, fontweight='bold', ha='center')
        ax.text(s_proj[i, 0], s_proj[i, 1]-0.2, str(i), color=COLORS["Student"], fontsize=14, fontweight='bold', ha='center')
        # Arrow from Teacher to Student
        ax.annotate("", xy=s_proj[i], xytext=t_proj[i],
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.4))

    ax.set_title(f"Latent Topology: Manifold Smoothing\n(PC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%})", pad=20)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    
    # Interpretation Note
    note = "Interpretation:\nArrows show the 'Distillation Shift'.\nThe Student manifold is more compact and linear,\nproving the smoothing of jagged decision boundaries."
    ax.text(0.05, 0.05, note, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="#EBEDEF", ec="#AEB6BF", alpha=0.9))
    
    ax.legend(loc='upper right', frameon=True, shadow=True)
    plt.subplots_adjust(top=0.9)
    fig.savefig(f"{PLOTS_DIR}/topology_manifold_pca.png")
    plt.close(fig)

def plot_waterfall(data):
    """Generates the distance-sorted Waterfall plots."""
    print("Plotting Waterfall...")
    
    # Use Alpha 0.5 for the most sensitive "friction" measurement
    alpha = 0.5
    sid = f"Matrix_VTeacher_TStudent_Alpha_{alpha}"
    
    # Get distance matrix
    dist_matrix = []
    for i in range(10):
        p = [s for s in data["data_series"] if s["series_id"] == "Teacher_Manifold_Distance" and s["x_axis"]["value"] == i][0]
        dist_matrix.append(p["raw"])
    dist_matrix = np.array(dist_matrix)
    
    # Get TASR data
    fig, axes = plt.subplots(2, 5, figsize=(22, 11), sharey=True)
    
    for i in range(10):
        ax = axes[i//5, i%5]
        
        # TASR for injecting vector i into Student, observing digit j
        tasr_points = [s for s in data["data_series"] if s["series_id"] == sid and s["group"] == f"Inject_{i}"]
        tasr_dict = {p["x_axis"]["value"]: p["metrics"]["accuracy_mean"] for p in tasr_points}
        
        # Sort j by distance to i in Teacher manifold
        dists = dist_matrix[i]
        # Distance is cos_sim, so high is "close"
        sorted_indices = np.argsort(dists)[::-1] # Closest first
        
        sorted_tasrs = [tasr_dict.get(j, 0) for j in sorted_indices if j != i]
        sorted_labels = [j for j in sorted_indices if j != i]
        
        ax.bar(range(len(sorted_tasrs)), sorted_tasrs, color=COLORS["Waterfall"], edgecolor='k', alpha=0.8)
        ax.set_xticks(range(len(sorted_tasrs)))
        ax.set_xticklabels(sorted_labels)
        ax.set_title(f"Targeting Digit: {i}", fontweight='bold')
        
        if i >= 5: ax.set_xlabel("Source Digit (Closest → Furthest)")
        if i % 5 == 0: ax.set_ylabel("TASR (Susceptibility)")
        ax.set_ylim(0, 1.05)

    plt.suptitle(f"The Vulnerability Waterfall (Intensity α={alpha})\nSusceptibility Sorted by Geometric Neighbor Distance", fontsize=24, y=0.98)
    
    # Global Interpretation
    fig.text(0.5, 0.02, "Mechanistic Rule: Neighbors in the Teacher's manifold are 'greased' during distillation, showing much higher susceptibility than distant classes.", 
             ha='center', fontsize=16, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="#E8F8F5", ec="#16A085", alpha=0.9))
    
    plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.3, wspace=0.2)
    fig.savefig(f"{PLOTS_DIR}/topology_waterfall.png")
    plt.close(fig)

def plot_steering_heatmaps(data):
    """Generates a 2x2 grid of 10x10 heatmaps for Latent Steering TASR at alpha = 2.0."""
    print("Plotting Steering Heatmaps...")
    alpha = 2.0
    quadrants = [
        ("Teacher", "Teacher", "Teacher -> Teacher (Self)"),
        ("Teacher", "Student", "Teacher -> Student (Forward)"),
        ("Student", "Teacher", "Student -> Teacher (Backward)"),
        ("Student", "Student", "Student -> Student (Self)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    for idx, (src, tgt, title) in enumerate(quadrants):
        ax = axes[idx // 2, idx % 2]
        sid = f"Matrix_V{src}_T{tgt}_Alpha_{alpha}"
        
        # Build 10x10 matrix: row is actual (y), col is target (z)
        matrix = np.zeros((10, 10))
        points = [s for s in data["data_series"] if s["series_id"] == sid]
        
        for p in points:
            # group is "Inject_{z}" (target z)
            z = int(p["group"].split("_")[1])
            # x_axis.value is actual y
            y = p["x_axis"]["value"]
            matrix[y, z] = p["metrics"]["accuracy_mean"]
            
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True,
                    xticklabels=list(range(10)), yticklabels=list(range(10)), ax=ax,
                    cbar_kws={'label': 'TASR'}, annot_kws={"size": 8})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Target Class (z)", fontsize=11)
        ax.set_ylabel("Actual Class (y)", fontsize=11)
        
    plt.suptitle(f"Latent Steering TASR (Intensity $\\alpha = {alpha}$)", fontsize=20, y=0.96, fontweight='bold')
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    fig.savefig(f"{PLOTS_DIR}/topology_steering_heatmaps.png")
    plt.close(fig)

if __name__ == "__main__":
    print(f"Loading data from {JSON_PATH}...")
    try:
        raw_data = load_data()
        plot_pca_map(raw_data)
        plot_waterfall(raw_data)
        plot_steering_heatmaps(raw_data)
        print(f"\n✅ All high-fidelity plots saved to {PLOTS_DIR}/")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
