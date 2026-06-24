import json
import numpy as np
import pandas as pd
from scipy import stats

def print_metrics():
    # Load primary data
    with open('/home/eran.b/takehome/outputs/latent_steering_attacks.json', 'r') as f:
        data = json.load(f)
    
    series = data['data_series']
    epsilons = [0.1, 0.3, 0.5]
    quadrants = [
        ("Teacher", "Teacher"),
        ("Teacher", "Student"),
        ("Student", "Teacher"),
        ("Student", "Student")
    ]
    
    print("=== Attack 1: USR Sweep ===")
    for src, tgt in quadrants:
        line = f"V{src} -> T{tgt}:"
        for eps in epsilons:
            sid = f"Attack1_USR_V{src}_T{tgt}_Epsilon"
            vals = [p['metrics']['accuracy_mean'] for p in series if p['series_id'] == sid and abs(p['x_axis']['value'] - eps) < 1e-5]
            mean_val = np.mean(vals) * 100.0 if vals else 0.0
            line += f" eps={eps}:{mean_val:.2f}% |"
        print(line)
        
    print("\n=== Attack 1: TSR Sweep ===")
    for src, tgt in quadrants:
        line = f"V{src} -> T{tgt}:"
        for eps in epsilons:
            sid = f"Attack1_TSR_V{src}_T{tgt}_Epsilon"
            vals = [p['metrics']['accuracy_mean'] for p in series if p['series_id'] == sid and abs(p['x_axis']['value'] - eps) < 1e-5]
            mean_val = np.mean(vals) * 100.0 if vals else 0.0
            line += f" eps={eps}:{mean_val:.2f}% |"
        print(line)

    print("\n=== Attack 2: USR Sweep ===")
    for src, tgt in quadrants:
        line = f"V{src} -> T{tgt}:"
        for eps in epsilons:
            sid = f"Attack2_USR_V{src}_T{tgt}_Epsilon"
            vals = [p['metrics']['accuracy_mean'] for p in series if p['series_id'] == sid and abs(p['x_axis']['value'] - eps) < 1e-5]
            mean_val = np.mean(vals) * 100.0 if vals else 0.0
            line += f" eps={eps}:{mean_val:.2f}% |"
        print(line)
        
    print("\n=== Attack 2: TSR Sweep ===")
    for src, tgt in quadrants:
        line = f"V{src} -> T{tgt}:"
        for eps in epsilons:
            sid = f"Attack2_TSR_V{src}_T{tgt}_Epsilon"
            vals = [p['metrics']['accuracy_mean'] for p in series if p['series_id'] == sid and abs(p['x_axis']['value'] - eps) < 1e-5]
            mean_val = np.mean(vals) * 100.0 if vals else 0.0
            line += f" eps={eps}:{mean_val:.2f}% |"
        print(line)

    # Load scatter data
    with open('/home/eran.b/takehome/outputs/latent_steering_scatter.json', 'r') as f:
        scatter_data = json.load(f)
        
    df = pd.DataFrame(scatter_data)
    if not df.empty:
        print("\n=== Table 4: Correlations (eps=0.3) ===")
        for att_type in [1, 2]:
            print(f"\nAttack {att_type}:")
            sub_df = df[df["attack_type"] == att_type]
            for src, tgt in quadrants:
                quad_key = f"V{src}_T{tgt}"
                quad_df = sub_df[sub_df["quadrant"] == quad_key]
                if quad_df.empty:
                    print(f"  {quad_key}: NO DATA")
                    continue
                x_vals = quad_df["latent_metric"].values
                y_vals = quad_df["confidence_drop"].values
                
                # Pearson
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                r_sq = r_value ** 2
                
                # Spearman
                spearman_rho, spearman_p = stats.spearmanr(x_vals, y_vals)
                
                print(f"  {quad_key} -> Pearson R^2: {r_sq:.3f}, Pearson R: {r_value:+.3f}, Spearman rho: {spearman_rho:+.3f}")

if __name__ == '__main__':
    print_metrics()
