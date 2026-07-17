"""
Plot results from same_init adversarial attacks experiment.
Reads: outputs/same_init_attacks_pretrained_matrices.json
Produces:
  - 10x10 TSR heatmaps per quadrant per epsilon per threat model
  - Bar chart of mean TSR per quadrant per epsilon
All output filenames prefixed with same_init_
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = "/home/eran.b/takehome/outputs/same_init_attacks_pretrained_matrices.json"
PLOTS_DIR    = "/home/eran.b/takehome/plots_a"
os.makedirs(PLOTS_DIR, exist_ok=True)

QUADRANT_LABELS = {
    "T_to_T": "Teacher → Teacher",
    "T_to_S": "Teacher → Student",
    "S_to_T": "Student → Teacher",
    "S_to_S": "Student → Student",
}
DIGIT_LABELS = [str(i) for i in range(10)]


def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def plot_heatmaps(data, threat_key, threat_label, eps_key, eps_val):
    quadrant_keys = ["T_to_T", "T_to_S", "S_to_T", "S_to_S"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"{threat_label}\n$\\epsilon = {eps_val}$  |  Strict Intersection Filter",
        fontsize=15, fontweight="bold", y=1.01
    )

    for ax, q_name in zip(axes.flatten(), quadrant_keys):
        matrix = np.array(data["results"][eps_key][q_name][threat_key])
        mask = np.eye(10, dtype=bool)
        sns.heatmap(
            matrix,
            ax=ax,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            xticklabels=DIGIT_LABELS,
            yticklabels=DIGIT_LABELS,
            cbar_kws={"label": "TSR"},
        )
        ax.set_title(QUADRANT_LABELS[q_name], fontweight="bold")
        ax.set_xlabel("Target Class")
        ax.set_ylabel("True Class")

    plt.tight_layout()
    fname = f"same_init_attacks_{threat_key}_heatmap_{eps_key}.png"
    fpath = os.path.join(PLOTS_DIR, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_mean_tsr_bars(data):
    quadrant_keys = list(QUADRANT_LABELS.keys())
    eps_keys = list(data["results"].keys())
    epsilons = data["metadata"]["epsilons"]

    for threat_key, threat_label in [("tsr_pgd", "Targeted PGD"), ("tsr_latent", "Latent Matching")]:
        fig, axes = plt.subplots(1, len(eps_keys), figsize=(5 * len(eps_keys), 5), sharey=True)
        if len(eps_keys) == 1:
            axes = [axes]

        for ax, (eps_key, eps_val) in zip(axes, zip(eps_keys, epsilons)):
            means = [
                data["results"][eps_key][q][f"{threat_key}_mean"]
                for q in quadrant_keys
            ]
            colors = ["#2196F3", "#FF5722", "#FF5722", "#4CAF50"]
            bars = ax.bar(
                [QUADRANT_LABELS[q] for q in quadrant_keys],
                means,
                color=colors,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.7,
            )
            ax.set_ylim(0, 1.05)
            ax.set_title(f"$\\epsilon = {eps_val}$", fontsize=13)
            ax.set_ylabel("Mean TSR" if ax == axes[0] else "")
            ax.set_xticks(range(len(quadrant_keys)))
            ax.set_xticklabels(
                [QUADRANT_LABELS[q] for q in quadrant_keys],
                rotation=30, ha="right", fontsize=9
            )
            ax.yaxis.grid(True, alpha=0.3)
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9
                )

        fig.suptitle(f"Mean TSR — {threat_label}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fname = f"same_init_attacks_{threat_key}_bar_summary.png"
        fpath = os.path.join(PLOTS_DIR, fname)
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


def main():
    print(f"Loading results from: {RESULTS_PATH}")
    data = load_results()

    meta = data["metadata"]
    print(f"  Teacher Acc: {meta['teacher_acc']:.3f}")
    print(f"  Student Acc: {meta['student_acc']:.3f}")
    print(f"  Joint Correct Fraction: {meta['joint_correct_fraction']:.3f}")

    eps_keys = list(data["results"].keys())
    epsilons = meta["epsilons"]

    for eps_key, eps_val in zip(eps_keys, epsilons):
        print(f"\nPlotting heatmaps for epsilon={eps_val}...")
        plot_heatmaps(data, "tsr_pgd",    "Targeted PGD",                  eps_key, eps_val)
        plot_heatmaps(data, "tsr_latent", "Latent Representation Matching", eps_key, eps_val)

    print("\nPlotting summary bar charts...")
    plot_mean_tsr_bars(data)

    print("\n✅ All plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
