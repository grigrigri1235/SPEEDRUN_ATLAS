import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    
    suffix = "pilot" if args.pilot else "full"
    json_path = f"outputs/boundary_attack_{suffix}.json"
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    matrices = {
        "Distance_Teacher": np.zeros((10, 10)),
        "Distance_Student": np.zeros((10, 10)),
        "Transfer_T2S": np.zeros((10, 10)),
        "Transfer_S2T": np.zeros((10, 10)),
        "Transfer_T2T": np.zeros((10, 10)),
        "Transfer_S2S": np.zeros((10, 10)),
        "Latent_Traversed_Teacher": np.zeros((10, 10)),
        "Latent_Traversed_Student": np.zeros((10, 10)),
        "Latent_Analytical_Teacher": np.zeros((10, 10)),
        "Latent_Analytical_Student": np.zeros((10, 10)),
    }
    
    for series in data.get("data_series", []):
        sid = series.get("series_id", "")
        src_d = int(series.get("group").split("_")[-1])
        tgt_d = series["x_axis"]["value"]
        mean_val = series["metrics"]["accuracy_mean"]
        
        if sid == "Boundary_Distance_VTeacher":
            matrices["Distance_Teacher"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Distance_VStudent":
            matrices["Distance_Student"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Transfer_VTeacher_TStudent":
            matrices["Transfer_T2S"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Transfer_VStudent_TTeacher":
            matrices["Transfer_S2T"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Transfer_VTeacher_TTeacher":
            matrices["Transfer_T2T"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Transfer_VStudent_TStudent":
            matrices["Transfer_S2S"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Latent_Distance_Traversed_VTeacher":
            matrices["Latent_Traversed_Teacher"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Latent_Distance_Traversed_VStudent":
            matrices["Latent_Traversed_Student"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Latent_Distance_Analytical_VTeacher":
            matrices["Latent_Analytical_Teacher"][src_d, tgt_d] = mean_val
        elif sid == "Boundary_Latent_Distance_Analytical_VStudent":
            matrices["Latent_Analytical_Student"][src_d, tgt_d] = mean_val

    # Mask diagonal
    for k in matrices:
        np.fill_diagonal(matrices[k], np.nan)
        
    # Calculate shared limits for Input Distance
    vmin_dist = min(np.nanmin(matrices["Distance_Teacher"]), np.nanmin(matrices["Distance_Student"]))
    vmax_dist = max(np.nanmax(matrices["Distance_Teacher"]), np.nanmax(matrices["Distance_Student"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sns.heatmap(matrices["Distance_Teacher"], annot=True, fmt=".2f", ax=axes[0,0], cmap="viridis_r", vmin=vmin_dist, vmax=vmax_dist)
    axes[0,0].set_title("Average Boundary Distance (Teacher)")
    
    sns.heatmap(matrices["Distance_Student"], annot=True, fmt=".2f", ax=axes[0,1], cmap="viridis_r", vmin=vmin_dist, vmax=vmax_dist)
    axes[0,1].set_title("Average Boundary Distance (Student)")
    
    sns.heatmap(matrices["Transfer_T2S"], annot=True, fmt=".2f", ax=axes[1,0], cmap="Reds", vmin=0, vmax=1)
    axes[1,0].set_title("Transfer Success: Teacher -> Student")
    
    sns.heatmap(matrices["Transfer_S2T"], annot=True, fmt=".2f", ax=axes[1,1], cmap="Reds", vmin=0, vmax=1)
    axes[1,1].set_title("Transfer Success: Student -> Teacher")
    
    for ax in axes.flatten():
        ax.set_xlabel("Target Digit")
        ax.set_ylabel("Source Digit")
        
    plt.tight_layout()
    os.makedirs("plots_a", exist_ok=True)
    plt.savefig(f"plots_a/boundary_attack_{suffix}.png", dpi=300)
    print(f"Saved plots_a/boundary_attack_{suffix}.png")
    plt.close()

    # Latent Space Plot
    # Calculate shared limits for Latent Traversed and Latent Analytical
    vmin_lt = min(np.nanmin(matrices["Latent_Traversed_Teacher"]), np.nanmin(matrices["Latent_Traversed_Student"]))
    vmax_lt = max(np.nanmax(matrices["Latent_Traversed_Teacher"]), np.nanmax(matrices["Latent_Traversed_Student"]))
    
    vmin_la = min(np.nanmin(matrices["Latent_Analytical_Teacher"]), np.nanmin(matrices["Latent_Analytical_Student"]))
    vmax_la = max(np.nanmax(matrices["Latent_Analytical_Teacher"]), np.nanmax(matrices["Latent_Analytical_Student"]))

    fig_latent, axes_latent = plt.subplots(2, 2, figsize=(14, 12))
    
    sns.heatmap(matrices["Latent_Traversed_Teacher"], annot=True, fmt=".2f", ax=axes_latent[0,0], cmap="plasma_r", vmin=vmin_lt, vmax=vmax_lt)
    axes_latent[0,0].set_title("Avg Traversed Latent Distance (Teacher)")
    
    sns.heatmap(matrices["Latent_Traversed_Student"], annot=True, fmt=".2f", ax=axes_latent[0,1], cmap="plasma_r", vmin=vmin_lt, vmax=vmax_lt)
    axes_latent[0,1].set_title("Avg Traversed Latent Distance (Student)")
    
    sns.heatmap(matrices["Latent_Analytical_Teacher"], annot=True, fmt=".2f", ax=axes_latent[1,0], cmap="plasma_r", vmin=vmin_la, vmax=vmax_la)
    axes_latent[1,0].set_title("Avg Analytical Latent Distance (Teacher)")
    
    sns.heatmap(matrices["Latent_Analytical_Student"], annot=True, fmt=".2f", ax=axes_latent[1,1], cmap="plasma_r", vmin=vmin_la, vmax=vmax_la)
    axes_latent[1,1].set_title("Avg Analytical Latent Distance (Student)")
    
    for ax in axes_latent.flatten():
        ax.set_xlabel("Target Digit")
        ax.set_ylabel("Source Digit")
        
    plt.tight_layout()
    plt.savefig(f"plots_a/boundary_attack_latent_{suffix}.png", dpi=300)
    print(f"Saved plots_a/boundary_attack_latent_{suffix}.png")
    plt.close()
    
    # Stats
    dT = matrices["Distance_Teacher"][~np.isnan(matrices["Distance_Teacher"])]
    dS = matrices["Distance_Student"][~np.isnan(matrices["Distance_Student"])]
    print(f"Mean dT = {dT.mean():.4f}, Mean dS = {dS.mean():.4f}")
    
    t2s = matrices["Transfer_T2S"][~np.isnan(matrices["Transfer_T2S"])]
    s2t = matrices["Transfer_S2T"][~np.isnan(matrices["Transfer_S2T"])]
    print(f"Mean Transfer T->S = {t2s.mean():.4f}")
    print(f"Mean Transfer S->T = {s2t.mean():.4f}")

    # Latent Stats
    ltT = matrices["Latent_Traversed_Teacher"][~np.isnan(matrices["Latent_Traversed_Teacher"])]
    ltS = matrices["Latent_Traversed_Student"][~np.isnan(matrices["Latent_Traversed_Student"])]
    laT = matrices["Latent_Analytical_Teacher"][~np.isnan(matrices["Latent_Analytical_Teacher"])]
    laS = matrices["Latent_Analytical_Student"][~np.isnan(matrices["Latent_Analytical_Student"])]
    print(f"Mean Latent Traversed: dT = {ltT.mean():.4f}, dS = {ltS.mean():.4f}")
    print(f"Mean Latent Analytical: dT = {laT.mean():.4f}, dS = {laS.mean():.4f}")

if __name__ == "__main__":
    main()
