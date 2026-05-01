"""
scratch/plot_weight_change_var.py
Generates Graph 4c: Ghost Channel Weight-Change Variance (GSNR Proof)
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR   = "/home/eran.b/takehome/outputs"
PLOTS_DIR     = "/home/eran.b/takehome/plots_a"
GRAPHS_DIR    = "/home/eran.b/takehome/graphs__std_a"
JSON_FILE     = os.path.join(OUTPUTS_DIR, "dropout_15e_stage.json")
OUT_VAR_PDF   = os.path.join(GRAPHS_DIR, "4c_dropout_weight_var_sweep.pdf")
OUT_VAR_PNG   = os.path.join(PLOTS_DIR, "4c_dropout_weight_var_sweep.png")

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
    'legend.fontsize':  12,
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

def extract_series(data, series_id):
    out = {}
    for pt in data.get("data_series", []):
        if pt["series_id"] == series_id:
            g = pt["group"]
            if g not in out:
                out[g] = []
            out[g].append((
                pt["x_axis"]["value"],
                pt["metrics"]["accuracy_mean"],
                pt["metrics"]["accuracy_std"],
                pt.get("raw", [])
            ))
    for g in out:
        out[g].sort(key=lambda r: r[0])
    return out

def main():
    with open(JSON_FILE) as f:
        data = json.load(f)

    # Use the direct per-sample GSNR = ||E[grad]||^2 / Var(grad)
    # now computed inside the sweep script and logged as Ghost_GSNR
    gsnr_raw = extract_series(data, "Ghost_GSNR")

    BATCH_SIZE = 512  # scale per-sample GSNR → batch GSNR

    fig, ax = plt.subplots(figsize=(6, 5))

    for group in ["Student-Only", "Teacher-Only", "Both"]:
        if group not in gsnr_raw:
            continue
        pts = gsnr_raw[group]
        xs    = np.array([p[0] for p in pts])
        means = np.array([p[1] for p in pts]) * BATCH_SIZE
        stds  = np.array([p[2] for p in pts]) * BATCH_SIZE
        color  = COLOR_MAP[group]
        marker = MARKER_MAP[group]
        ax.plot(xs, means, marker=marker, label=group, color=color)

        lower_bound = np.maximum(0, means - stds)
        ax.fill_between(xs, lower_bound, means + stds, color=color, alpha=0.2)

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Absolute Noise Floor (Zero Signal)')
    # Add annotation for the bias floor
    ax.text(0.06, 1.2, "Mathematical Noise Floor (Bias)", color='red', fontsize=10, fontweight='bold', alpha=0.8)
    ax.set_title("Ghost Batch GSNR (Init) | 1.0 = Zero Signal Floor", pad=15)
    ax.set_yscale("log")
    ax.set_xlabel("Dropout Probability (p)")
    ax.set_ylabel(r"Batch GSNR = $B \cdot \|\mathbb{E}[\nabla L]\|^2 / \mathrm{Var}(\nabla L)$")
    ax.legend()
    ax.set_xlim(0.05, 0.55)
    plt.tight_layout()

    plt.savefig(OUT_VAR_PDF)
    plt.savefig(OUT_VAR_PNG)
    plt.close()
    print(f"Saved: {OUT_VAR_PDF}")
    print(f"Saved: {OUT_VAR_PNG}")

    print("\n--- GSNR DATA FOR REPORT ---")
    for group in ["Student-Only", "Teacher-Only", "Both"]:
        if group not in gsnr_raw:
            continue
        for x, m, s, raw in gsnr_raw[group]:
            print(f"  {group}  p={x}: Batch GSNR={m*BATCH_SIZE:.1f} ± {s*BATCH_SIZE:.1f}")

if __name__ == "__main__":
    main()
