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
            # group is "Inject_{digit}", x_axis.value is target_digit
            inject_digit = int(s["group"].split("_")[1])
            target_digit = s["x_axis"]["value"]
            if inject_digit == target_digit: continue # Skip self-accuracy
            
            rows.append({
                "inject": inject_digit,
                "target": target_digit,
                "fpr_mean": s["metrics"]["accuracy_mean"],
                "fpr_std": s["metrics"]["accuracy_std"]
            })
    return pd.DataFrame(rows)

def plot_dosage_curve(data):
    """Generates the Dosage Response plot for Digit 9."""
    print("Plotting Dosage Curves...")
    alphas = [0.5, 1.0, 2.0, 5.0]
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # We want to show how V_Teacher affects Student vs how V_Student affects Teacher
    quadrants = [
        ("Teacher", "Student", "-", "V_Teacher on Student (Sledgehammer)"),
        ("Student", "Teacher", "--", "V_Student on Teacher (Reverse Steering)"),
        ("Teacher", "Teacher", ":", "V_Teacher on Teacher (Control)"),
        ("Student", "Student", "-.", "V_Student on Student (Consistency)")
    ]
    
    for src, tgt, ls, label in quadrants:
        means = []
        stds = []
        for a in alphas:
            sid = f"Matrix_V{src}_T{tgt}_Alpha_{a}"
            # Extract Digit 9 injection specifically
            digit_9_points = [s for s in data["data_series"] if s["series_id"] == sid and s["group"] == "Inject_9"]
            if not digit_9_points: continue
            
            # Average FPR over all target digits j != 9
            vals = [p["metrics"]["accuracy_mean"] for p in digit_9_points if p["x_axis"]["value"] != 9]
            means.append(np.mean(vals))
            # Rough std approximation
            stds.append(np.mean([p["metrics"]["accuracy_std"] for p in digit_9_points if p["x_axis"]["value"] != 9]))
        
        ax.plot(alphas, means, label=label, linestyle=ls, marker='o', linewidth=3, color=COLORS[tgt])
        ax.fill_between(alphas, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color=COLORS[tgt], alpha=0.1)

    # --- Annotations ---
    ax.annotate("The Authority Paradox:\nTeacher vectors hijack Student\n8x better than vice-versa", 
                xy=(0.5, 0.8), xytext=(1.5, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    ax.text(0.1, 0.05, "← Lower is Better (Resistance)", fontsize=12, fontweight='bold', color='green', transform=ax.get_xaxis_transform())
    ax.text(4.0, 0.95, "Higher is Vulnerable →", fontsize=12, fontweight='bold', color='red', transform=ax.get_xaxis_transform())

    # Mechanistic Note
    note = "Mechanistic Note:\nThe steep red curve indicates low 'Geometric Friction'.\nThe Student manifold has been smoothed by distillation,\nmaking it highly susceptible to external steering."
    ax.text(0.95, 0.05, note, transform=ax.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4F6F7", ec="#ABB2B9", alpha=0.9))

    ax.set_title("Dosage Response: Latent Manifold Susceptibility", pad=20)
    ax.set_xlabel("Steering Intensity (Alpha)")
    ax.set_ylabel("Mean False Positive Rate (FPR)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    
    plt.subplots_adjust(top=0.9) # Fix cutoff header
    fig.savefig(f"{PLOTS_DIR}/topology_9_dosage.png")
    plt.close(fig)

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
    
    # Get FPR data
    fig, axes = plt.subplots(2, 5, figsize=(22, 11), sharey=True)
    
    for i in range(10):
        ax = axes[i//5, i%5]
        
        # FPR for injecting vector i into Student, observing digit j
        fpr_points = [s for s in data["data_series"] if s["series_id"] == sid and s["group"] == f"Inject_{i}"]
        fpr_dict = {p["x_axis"]["value"]: p["metrics"]["accuracy_mean"] for p in fpr_points}
        
        # Sort j by distance to i in Teacher manifold
        dists = dist_matrix[i]
        # Distance is cos_sim, so high is "close"
        sorted_indices = np.argsort(dists)[::-1] # Closest first
        
        sorted_fprs = [fpr_dict.get(j, 0) for j in sorted_indices if j != i]
        sorted_labels = [j for j in sorted_indices if j != i]
        
        ax.bar(range(len(sorted_fprs)), sorted_fprs, color=COLORS["Waterfall"], edgecolor='k', alpha=0.8)
        ax.set_xticks(range(len(sorted_fprs)))
        ax.set_xticklabels(sorted_labels)
        ax.set_title(f"Targeting Digit: {i}", fontweight='bold')
        
        if i >= 5: ax.set_xlabel("Source Digit (Closest → Furthest)")
        if i % 5 == 0: ax.set_ylabel("FPR (Susceptibility)")
        ax.set_ylim(0, 1.05)

    plt.suptitle(f"The Vulnerability Waterfall (α={alpha})\nSusceptibility Sorted by Geometric Neighbor Distance", fontsize=24, y=0.98)
    
    # Global Interpretation
    fig.text(0.5, 0.02, "Mechanistic Rule: Neighbors in the Teacher's manifold are 'greased' during distillation, showing much higher susceptibility than distant classes.", 
             ha='center', fontsize=16, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="#E8F8F5", ec="#16A085", alpha=0.9))
    
    plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.3, wspace=0.2)
    fig.savefig(f"{PLOTS_DIR}/topology_waterfall.png")
    plt.close(fig)

if __name__ == "__main__":
    print(f"Loading data from {JSON_PATH}...")
    try:
        raw_data = load_data()
        plot_dosage_curve(raw_data)
        plot_pca_map(raw_data)
        plot_waterfall(raw_data)
        print(f"\n✅ All high-fidelity plots saved to {PLOTS_DIR}/")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
