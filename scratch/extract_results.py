import json

with open('outputs/l1_analysis_v2_results.json', 'r') as f:
    data = json.load(f)

print("| Regime | lambda | Student Acc | Layer 0 Sim | Layer 1 Sim | Layer 2 Sim |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")

# Baselines
def get_base(name):
    return f"{data['baselines'][name]['accuracy_mean']:.3f}"

print(f"| Baseline | 0 | {get_base('No_Reg_Student_MNIST')} | {get_base('No_Reg_Layer0_Cosine_Sim')} | {get_base('No_Reg_Layer1_Cosine_Sim')} | {get_base('No_Reg_Layer2_Cosine_Sim')} |")

# Collect series
series = {}
for entry in data['data_series']:
    key = (entry['group'], entry['x_axis']['value'])
    if key not in series: series[key] = {}
    series[key][entry['series_id']] = entry['metrics']['accuracy_mean']

for (group, lam), metrics in sorted(series.items()):
    acc = metrics.get('Student_MNIST_Accuracy', 0)
    s0 = metrics.get('Layer0_Cosine_Sim', 0)
    s1 = metrics.get('Layer1_Cosine_Sim', 0)
    s2 = metrics.get('Layer2_Cosine_Sim', 0)
    print(f"| {group} | {lam} | {acc:.3f} | {s0:.3f} | {s1:.3f} | {s2:.3f} |")
