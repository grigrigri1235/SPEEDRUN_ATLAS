import json
import numpy as np
from scipy import stats

with open('/home/eran.b/takehome/outputs/latent_steering_scatter.json', 'r') as f:
    scatter_data = json.load(f)

quadrant_labels = {
    "VTeacher_TTeacher": "Teacher -> Teacher (Self)",
    "VTeacher_TStudent": "Teacher -> Student (Distill Transfer)",
    "VStudent_TTeacher": "Student -> Teacher (Backward Transfer)",
    "VStudent_TStudent": "Student -> Student (Self)"
}

print("=== SCATTER DATA CORRELATION ANALYSIS (R² VALUES) ===")
for att_type in [1, 2]:
    att_name = "Attack 1: Input PGD (ε = 0.1)" if att_type == 1 else "Attack 2: Latent Steering (α = 1.0)"
    print(f"\n{att_name}")
    
    # Filter by attack type
    att_data = [d for d in scatter_data if d["attack_type"] == att_type]
    
    for quad in ["VTeacher_TTeacher", "VTeacher_TStudent", "VStudent_TTeacher", "VStudent_TStudent"]:
        quad_data = [d for d in att_data if d["quadrant"] == quad]
        if not quad_data:
            print(f"  {quadrant_labels[quad]}: No data")
            continue
            
        x_vals = [d["latent_metric"] for d in quad_data]
        y_vals = [d["confidence_drop"] for d in quad_data]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_sq = r_value ** 2
        print(f"  {quadrant_labels[quad]:<40}: R² = {r_sq:.5f} | slope = {slope:.5f} | p-value = {p_value:.5e}")
