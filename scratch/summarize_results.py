import json
import numpy as np

with open('/home/eran.b/takehome/outputs/latent_steering_attacks.json', 'r') as f:
    data = json.load(f)

print("=== BASELINES ===")
for k, v in data['baselines'].items():
    print(f"{k}: Mean={v['accuracy_mean']:.4f}, Std={v['accuracy_std']:.4f}")

# Group the data series by quadrant, attack type, and parameter value
results = {}

for series in data['data_series']:
    sid = series['series_id']
    if 'Confusion' in sid:
        continue
    
    # Example: Attack1_Accuracy_VTeacher_TTeacher_Epsilon
    # or Attack1_Latent_Shift_VTeacher_TTeacher_Epsilon
    parts = sid.split('_')
    attack_name = parts[0] # Attack1 or Attack2
    metric_name = parts[1] # Accuracy or Latent
    
    # quadrant is parts[2] (e.g. VTeacher) and parts[3] (e.g. TTeacher)
    quad = f"{parts[2]}_{parts[3]}"
    
    param_label = series['x_axis']['label']
    param_val = series['x_axis']['value']
    
    group_digit = series['group'] # e.g. Digit_0
    mean_val = series['metrics']['accuracy_mean'] # Wait, let's check key name. Wait, the metric key for all series is 'accuracy_mean' inside 'metrics' even if it's shift/distance!
    # Let's verify by printing metrics keys. Yes, in the logs we saw:
    # "metrics": { "accuracy_mean": 3.845618963241577, "accuracy_std": 0.11255805192602202 }
    # So both accuracy and latent metrics are keyed as accuracy_mean/accuracy_std.
    
    key = (attack_name, quad, metric_name, param_val)
    if key not in results:
        results[key] = []
    results[key].append(mean_val)

print("\n=== ATTACK PERFORMANCE (Averages across all 10 digits) ===")
# Sort and print
for key in sorted(results.keys()):
    attack_name, quad, metric_name, param_val = key
    vals = results[key]
    avg_val = np.mean(vals)
    std_val = np.std(vals)
    print(f"{attack_name} | {quad} | {metric_name} | {param_val} = {avg_val:.4f} (std across digits={std_val:.4f})")
