import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    json_path = "/home/eran.b/takehome/outputs/raz_steering.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Dictionary to hold the extracted arrays
    # format: metrics[series_id][digit] = (mean, std)
    metrics = {
        "LHS_Deviation": np.zeros((10, 2)),
        "RHS_Bound": np.zeros((10, 2)),
        "Rho": np.zeros((10, 2)),
        "CosSim_Actual": np.zeros((10, 2)),
        "CosSim_Lower_Bound": np.zeros((10, 2)),
    }

    # Extract data
    for series in data.get("data_series", []):
        sid = series.get("series_id")
        if sid in metrics:
            digit = series["x_axis"]["value"]
            metrics[sid][digit][0] = series["metrics"]["accuracy_mean"]
            metrics[sid][digit][1] = series["metrics"]["accuracy_std"]

    digits = np.arange(10)
    out_dir = "/home/eran.b/takehome/plots_report"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Plot Steering Vector Deviation vs Theoretical Upper Bound
    plt.figure(figsize=(10, 6))
    plt.bar(digits, metrics["LHS_Deviation"][:, 0], yerr=metrics["LHS_Deviation"][:, 1], 
            capsize=5, alpha=0.7, label='Actual Deviation', color='steelblue')
    
    # Plot black markers for RHS bounds
    # using a very short errorbar with no line, or a horizontal marker for each
    plt.scatter(digits, metrics["RHS_Bound"][:, 0], color='black', marker='_', s=500, 
                linewidth=3, label='Theoretical Upper Bound', zorder=5)

    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('L2 Deviation Magnitude', fontsize=12)
    plt.title('Steering Vector Deviation vs Theoretical Upper Bound', fontsize=14)
    plt.xticks(digits)
    plt.legend(loc='center right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path1 = os.path.join(out_dir, "LHS_vs_RHS_Bound.pdf")
    plt.savefig(out_path1, dpi=300, bbox_inches='tight')
    plt.savefig(out_path1.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path1}")

    # 2. Plot CosSim vs Lower Bound
    plt.figure(figsize=(10, 6))
    plt.bar(digits, metrics["CosSim_Actual"][:, 0], yerr=metrics["CosSim_Actual"][:, 1], 
            capsize=5, alpha=0.7, label='Actual Cosine Similarity', color='forestgreen')
    
    plt.scatter(digits, metrics["CosSim_Lower_Bound"][:, 0], color='black', marker='_', s=500, 
                linewidth=3, label='Theoretical Lower Bound', zorder=5)

    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Directional Alignment: Actual vs Theoretical Lower Bound', fontsize=14)
    plt.xticks(digits)
    # Adjust Y axis to show the similarities clearly
    # Cosine similarities are often close to 1
    plt.ylim(0, 1.1)
    plt.legend(loc='center right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path2 = os.path.join(out_dir, "CosSim_vs_Lower_Bound.pdf")
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.savefig(out_path2.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path2}")

if __name__ == "__main__":
    main()
