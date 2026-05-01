"""
scratch/plot_gsnr_trajectory.py
Generates Graph 4d: Ghost Channel GSNR Trajectory over Epochs
Uses Ghost_GSNR_Trajectory series from dropout_15e_stage.json
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR   = "/home/eran.b/takehome/outputs"
PLOTS_DIR     = "/home/eran.b/takehome/plots_a"
GRAPHS_DIR    = "/home/eran.b/takehome/graphs__std_a"
JSON_FILE     = os.path.join(OUTPUTS_DIR, "dropout_15e_stage.json")
OUT_PDF       = os.path.join(GRAPHS_DIR, "4d_dropout_gsnr_trajectory.pdf")
OUT_PNG       = os.path.join(PLOTS_DIR,  "4d_dropout_gsnr_trajectory.png")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# --- Aesthetic Standard (matching existing graphs) ---
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  10,
    'lines.linewidth':  2,
})

COLOR_MAP = {
    "Student-Only": "#2ca02c",
    "Teacher-Only": "#1f77b4",
    "Both":         "#d62728",
}
MARKER_MAP = {
    "Student-Only": "o",
    "Teacher-Only": "s",
    "Both":         "^",
}
LINESTYLE_MAP = {
    0.1: "-",
    0.3: "--",
    0.5: ":",
}

def extract_trajectory(data, series_id):
    """Extract Ghost_GSNR_Trajectory grouped by group (e.g. 'Student-Only_p0.5')."""
    out = {}
    for pt in data.get("data_series", []):
        if pt["series_id"] == series_id:
            g = pt["group"]
            out.setdefault(g, []).append((
                pt["x_axis"]["value"],
                pt["metrics"]["accuracy_mean"],
                pt["metrics"]["accuracy_std"],
            ))
    for g in out:
        out[g].sort(key=lambda r: r[0])
    return out


def main():
    with open(JSON_FILE) as f:
        data = json.load(f)

    traj = extract_trajectory(data, "Ghost_GSNR_Trajectory")

    fig, ax = plt.subplots(figsize=(8, 5))

    BATCH_SIZE = 512  # scale per-sample GSNR → batch GSNR

    for regime in ["Student-Only", "Teacher-Only", "Both"]:
        for p in [0.1, 0.3, 0.5]:
            key = f"{regime}_p{p}"
            if key not in traj:
                continue
            pts = traj[key]
            epochs = np.array([r[0] for r in pts])
            means  = np.array([r[1] for r in pts]) * BATCH_SIZE
            stds   = np.array([r[2] for r in pts]) * BATCH_SIZE

            color = COLOR_MAP[regime]
            marker = MARKER_MAP[regime]
            ls = LINESTYLE_MAP[p]
            label = f"{regime} p={p}"

            ax.plot(epochs, means, marker=marker, markersize=4, linestyle=ls,
                    color=color, label=label, alpha=0.9)
            lower = np.maximum(0, means - stds)
            ax.fill_between(epochs, lower, means + stds, color=color, alpha=0.08)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Absolute Noise Floor (Zero Signal)')
    ax.text(0.5, 1.2, "Mathematical Noise Floor (Bias)", color='red', fontsize=10, fontweight='bold', alpha=0.8)
    ax.set_title("Ghost Batch GSNR Trajectory During Distillation", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Batch GSNR = $B \cdot \|\mathbb{E}[\nabla L]\|^2 / \mathrm{Tr}(\mathrm{Cov}(\nabla L))$")
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_xlim(-0.5, 15.5)
    plt.tight_layout()

    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG)
    plt.close()
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
