import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_path = '/home/eran.b/takehome/outputs/convnext_baseline.json'

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    exit(1)

with open(json_path, 'r') as f:
    data = json.load(f)

series = data['data_series']

groups = []
teacher_means = {}
teacher_stds = {}
student_means = {}
student_stds = {}

n_models = data['metadata']['n_models']

for s in series:
    g = s['group']
    if g not in groups:
        groups.append(g)
    
    role = s['x_axis']['value']
    metrics = s['metrics']
    mean = metrics['accuracy_mean']
    ci_95 = metrics['accuracy_std'] / np.sqrt(n_models) * 1.96
    
    if role == "Teacher":
        teacher_means[g] = mean
        teacher_stds[g] = ci_95
    elif role == "Student":
        student_means[g] = mean
        student_stds[g] = ci_95

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

t_means = [teacher_means.get(g, 0) for g in groups]
t_stds = [teacher_stds.get(g, 0) for g in groups]

s_means = [student_means.get(g, 0) for g in groups]
s_stds = [student_stds.get(g, 0) for g in groups]

rects1 = ax.bar(x - width/2, t_means, width, yerr=t_stds, label='Teacher', capsize=5, color='#1f77b4', alpha=0.9)
rects2 = ax.bar(x + width/2, s_means, width, yerr=s_stds, label='Student', capsize=5, color='#1f77b4', alpha=0.4, hatch='//')

ax.axhline(0.10, ls='--', c='black', label='Chance Baseline (10%)')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Subliminal Transfer: MLP vs Hybrid Conv-MLP', fontsize=14)
ax.set_xticks(x)

import textwrap
wrapped_labels = [textwrap.fill(label, 15) for label in groups]
ax.set_xticklabels(wrapped_labels, rotation=0, fontsize=11)

ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
os.makedirs('/home/eran.b/takehome/plots_a', exist_ok=True)
output_path = '/home/eran.b/takehome/plots_a/convnext_baseline_results.png'
plt.savefig(output_path, dpi=150)
print(f"Plot successfully saved to {output_path}")
