import json
import numpy as np
import matplotlib.pyplot as plt
import os

# We need to recreate the plot_combined function logic but use the JSON data
def plot_combined_from_json(json_path, out_path, title):
    with open(json_path, 'r') as f:
        full_data = json.load(f)
    
    # UniLogger stores data in data_series
    series = full_data.get("data_series", [])
    
    # Regimes
    regimes = ['Student-Only', 'Teacher-Only', 'Both']
    regime_data = {r: {} for r in regimes}
    
    for item in series:
        regime = item["group"]
        lam = item["x_axis"]["value"]
        sid = item["series_id"]
        vals = item["raw"]
        
        if regime not in regime_data: continue
        if lam not in regime_data[regime]: regime_data[regime][lam] = {}
        
        regime_data[regime][lam][sid] = (np.mean(vals), np.std(vals))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    
    for i, regime in enumerate(regimes):
        ax = axes[i]
        lams = sorted(regime_data[regime].keys())
        if not lams: continue
        
        xs = np.array(lams)
        stud_m = np.array([regime_data[regime][l][ "Student_MNIST_Accuracy"][0] for l in lams])
        stud_s = np.array([regime_data[regime][l][ "Student_MNIST_Accuracy"][1] for l in lams])
        teach_m = np.array([regime_data[regime][l]["Teacher_MNIST_Accuracy"][0] for l in lams])
        teach_s = np.array([regime_data[regime][l]["Teacher_MNIST_Accuracy"][1] for l in lams])
        cosim_m = np.array([regime_data[regime][l]["Avg_Cosine_Similarity"][0] for l in lams])
        cosim_s = np.array([regime_data[regime][l]["Avg_Cosine_Similarity"][1] for l in lams])
        
        ax.plot(xs, stud_m, 'g-o', label='Student MNIST Acc')
        ax.fill_between(xs, stud_m-stud_s, stud_m+stud_s, alpha=0.15, color='g')
        ax.plot(xs, teach_m, 'b-s', label='Teacher MNIST Acc')
        ax.fill_between(xs, teach_m-teach_s, teach_m+teach_s, alpha=0.15, color='b')
        ax.plot(xs, cosim_m, 'r--^', label='Avg Cosine Sim')
        ax.fill_between(xs, cosim_m-cosim_s, cosim_m+cosim_s, alpha=0.15, color='r')
        
        ax.set_xscale('log')
        ax.set_title(regime)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_combined_from_json("outputs/l1_analysis_v4_results.json", "plots_a/l1_analysis_v4_combined.png", "L1 Analysis v4")
    plot_combined_from_json("outputs/l2_analysis_v1_results.json", "plots_a/l2_analysis_v1_combined.png", "L2 Analysis v1")
