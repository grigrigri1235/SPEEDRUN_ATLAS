import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup_plotting_style():
    """Apply premium aesthetics for NeurIPS-ready plots."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1
    })

def load_unified_json(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_experiment(data, output_dir):
    setup_plotting_style()
    metadata = data["metadata"]
    series_list = data["data_series"]
    baselines = data.get("baselines", {})

    if not series_list:
        print(f"Skipping {metadata['experiment_id']} - no data series found.")
        return

    # Convert series to DataFrame
    rows = []
    for point in series_list:
        rows.append({
            "Series": point["series_id"],
            "Group": point["group"],
            "X": point["x_axis"]["value"],
            "X_Label": point["x_axis"]["label"],
            "Accuracy": point["metrics"]["accuracy_mean"],
            "Std": point["metrics"]["accuracy_std"]
        })
    df = pd.DataFrame(rows)

    # Unique series IDs
    experiment_id = metadata["experiment_id"]
    
    unique_series = df["Series"].unique()
    
    for sid in unique_series:
        plt.figure(figsize=(8, 5))
        sub_df = df[df["Series"] == sid]
        
        # Line plot with markers
        sns.lineplot(
            data=sub_df, 
            x="X", y="Accuracy", hue="Group", 
            marker="o", markersize=6, linewidth=2
        )
        
        # Add error bars manually if needed (seaborn lineplot does confidence intervals if raw data provided, 
        # but here we have pre-computed means)
        for group in sub_df["Group"].unique():
            group_df = sub_df[sub_df["Group"] == group]
            plt.fill_between(
                group_df["X"], 
                group_df["Accuracy"] - group_df["Std"], 
                group_df["Accuracy"] + group_df["Std"], 
                alpha=0.15
            )

        # Baselines
        colors = sns.color_palette("rocket", len(baselines))
        for i, (name, bdata) in enumerate(baselines.items()):
            plt.axhline(
                bdata["accuracy_mean"], 
                ls="--", color=colors[i], label=f"Baseline: {name}", alpha=0.7
            )

        plt.title(f"{experiment_id}\nSeries: {sid}")
        plt.xlabel(sub_df["X_Label"].iloc[0])
        plt.ylabel("Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        save_path = os.path.join(output_dir, f"{experiment_id}_{sid}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  📈 Created plot: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS-standard plots from Uni-Code JSONs")
    parser.add_argument("--input", "-i", type=str, help="Input directory (outputs/) or specific JSON file")
    parser.add_argument("--outdir", "-o", type=str, default="plots_unified", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".json")]

    for fpath in files:
        print(f"Processing {fpath}...")
        try:
            data = load_unified_json(fpath)
            plot_experiment(data, args.outdir)
        except Exception as e:
            print(f"  ❌ Error processing {fpath}: {e}")

if __name__ == "__main__":
    main()
