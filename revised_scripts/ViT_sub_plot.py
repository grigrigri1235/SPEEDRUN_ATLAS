import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_path = 'outputs/ViT_sub.json'

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

series = data['data_series']
n_models = data['metadata']['n_models']

# We have 2 roles to plot on X-axis
roles = [
    "Teacher",
    "Student"
]

# We have 2 groups to compare
groups = ["MLP Baseline", "ViT"]

# Initialize nested dicts to hold means and CIs
means = {g: {r: 0.0 for r in roles} for g in groups}
cis = {g: {r: 0.0 for r in roles} for g in groups}

for s in series:
    g = s['group']
    r = s['x_axis']['value']
    if g in groups and r in roles:
        mean = s['metrics']['accuracy_mean']
        std = s['metrics']['accuracy_std']
        # 95% CI calculation
        ci = (std / np.sqrt(n_models)) * 1.96
        means[g][r] = mean
        cis[g][r] = ci

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

teacher_means = [means[g]["Teacher"] for g in groups]
teacher_cis = [cis[g]["Teacher"] for g in groups]

student_means = [means[g]["Student"] for g in groups]
student_cis = [cis[g]["Student"] for g in groups]

rects1 = ax.bar(x - width/2, teacher_means, width, yerr=teacher_cis, label='Teacher', capsize=5, color='#1f77b4', alpha=0.9)
rects2 = ax.bar(x + width/2, student_means, width, yerr=student_cis, label='Student', capsize=5, color='#ff7f0e', alpha=0.9)

ax.axhline(0.10, ls='--', c='black', label='Chance Baseline (10%)')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title("Subliminal Transfer: MLP vs ViT", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)

ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
os.makedirs('/home/eran.b/takehome/plots_a', exist_ok=True)
output_path = "/home/eran.b/takehome/plots_a/ViT_sub_results.png"
plt.savefig(output_path, dpi=150)
print(f"Plot successfully saved to {output_path}")
