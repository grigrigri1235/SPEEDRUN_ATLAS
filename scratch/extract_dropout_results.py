import json
import numpy as np

MASTER_FILE = "outputs/mechanism_sweep_results.json"
STAGE_FILE = "outputs/dropout_15e_stage.json"

with open(MASTER_FILE, "r") as f:
    master_data = json.load(f)

with open(STAGE_FILE, "r") as f:
    stage_data = json.load(f)

def get_series(series_id, group):
    res = {}
    for s in master_data["data_series"]:
        if s["series_id"] == series_id and s["group"] == group:
            res[s["x_axis"]["value"]] = s["metrics"]["accuracy_mean"]
    return res

regimes = ["Student-Only", "Teacher-Only"]
probs = [0.0, 0.1, 0.3, 0.5]

print("# Dropout Robustness Report (15 Epochs)")

for regime in regimes:
    print(f"\n### {regime} Dropout Analysis")
    print("| Prob (p) | **Student Acc** | **S ↔ T (Activ.)** | S ↔ Init (Act.) | T ↔ Init (Act.) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    base_acc = stage_data["baselines"]["No_Reg_Student_MNIST"]["accuracy_mean"]
    base_st = stage_data["baselines"]["No_Reg_Avg_Cosine_Sim"]["accuracy_mean"]
    
    accs = get_series("Dropout_Sweep", regime)
    st_sims = get_series("Avg_Cosine_Similarity", regime)
    si_sims = get_series("Student_vs_Init_Cosine_Sim", regime)
    ti_sims = get_series("Teacher_vs_Init_Cosine_Sim", regime)

    for p in probs:
        if p == 0.0:
            print(f"| **0 (Baseline)** | **{base_acc:.3f}** | **{base_st:.3f}** | 1.000 | {base_st:.3f} |")
            continue
        
        acc = accs.get(p, 0.0)
        st = st_sims.get(p, 0.0)
        si = si_sims.get(p, 0.0)
        ti = ti_sims.get(p, 0.0)
        print(f"| **{p}** | **{acc:.3f}** | **{st:.3f}** | {si:.3f} | {ti:.3f} |")
