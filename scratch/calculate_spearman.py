import json
import numpy as np
from scipy import stats

scatter_path = '/home/eran.b/takehome/outputs/latent_steering_scatter.json'
print(f"Loading data from {scatter_path}...")
with open(scatter_path, 'r') as f:
    scatter_data = json.load(f)

quadrant_labels = {
    "VTeacher_TTeacher": "Teacher -> Teacher (Self)",
    "VTeacher_TStudent": "Teacher -> Student (Distill Transfer)",
    "VStudent_TTeacher": "Student -> Teacher (Backward Transfer)",
    "VStudent_TStudent": "Student -> Student (Self)"
}

print("\n=== SCATTER DATA CORRELATION ANALYSIS (PEARSON & SPEARMAN) ===")
for att_type in [1, 2]:
    att_name = "Attack 1: Input PGD (ε = 0.1)" if att_type == 1 else "Attack 2: Latent Steering (α = 1.0)"
    print(f"\n{att_name}")
    print("-" * 100)
    print(f"{'Quadrant':<40} | {'Pearson R²':<12} | {'Pearson Slope':<14} | {'Spearman ρ':<12} | {'Spearman p-val':<15}")
    print("-" * 100)
    
    # Filter by attack type
    att_data = [d for d in scatter_data if d["attack_type"] == att_type]
    
    for quad in ["VTeacher_TTeacher", "VTeacher_TStudent", "VStudent_TTeacher", "VStudent_TStudent"]:
        quad_data = [d for d in att_data if d["quadrant"] == quad]
        if not quad_data:
            print(f"  {quadrant_labels[quad]:<40} | No data")
            continue
            
        x_vals = np.array([d["latent_metric"] for d in quad_data])
        y_vals = np.array([d["confidence_drop"] for d in quad_data])
        
        # Pearson linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_sq = r_value ** 2
        
        # Spearman correlation
        spearman_rho, spearman_pval = stats.spearmanr(x_vals, y_vals)
        
        print(f"  {quadrant_labels[quad]:<40} | {r_sq:10.5f} | {slope:12.5f} | {spearman_rho:10.5f} | {spearman_pval:13.5e}")
