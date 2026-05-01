import json
import numpy as np

def extract_triangle(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    ds = data.get("data_series", [])
    summary = {}
    
    for item in ds:
        sid = item["series_id"]
        regime = item["group"]
        lam = item["x_axis"]["value"]
        
        if sid not in ["Avg_Cosine_Similarity", "Student_vs_Init_Cosine_Sim", "Teacher_vs_Init_Cosine_Sim"]:
            continue
            
        vals = item["raw"]
        mean = np.mean(vals)
        std = np.std(vals)
        
        if regime not in summary:
            summary[regime] = {}
        if lam not in summary[regime]:
            summary[regime][lam] = {}
        
        summary[regime][lam][sid] = (mean, std)
            
    return summary

def print_triangle_table(summary, label):
    print(f"\n### {label} ###")
    regime = "Teacher-Only"
    lambdas = sorted(list(summary[regime].keys()))
    
    print(f"\nRegime: {regime} (The Stagnation Probe)")
    print(f"{'Lambda':<10} | {'S ↔ T':<12} | {'S ↔ Init':<12} | {'T ↔ Init':<12}")
    print("-" * 55)
    
    for lam in lambdas:
        row = summary[regime][lam]
        st = f"{row.get('Avg_Cosine_Similarity', (0,0))[0]:.3f}"
        si = f"{row.get('Student_vs_Init_Cosine_Sim', (0,0))[0]:.3f}"
        ti = f"{row.get('Teacher_vs_Init_Cosine_Sim', (0,0))[0]:.3f}"
        print(f"{lam:<10} | {st:<12} | {si:<12} | {ti:<12}")

if __name__ == "__main__":
    l1_sum = extract_triangle("outputs/l1_analysis_v5_results.json")
    l2_sum = extract_triangle("outputs/l2_analysis_v2_results.json")
    
    print_triangle_table(l1_sum, "L1 Triangle Analysis (v5)")
    print_triangle_table(l2_sum, "L2 Triangle Analysis (v2)")
