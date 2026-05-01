import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
with open('outputs/l1_analysis_v2_results.json', 'r') as f:
    data = json.load(f)

# Organize series
series_map = {}
for entry in data['data_series']:
    g = entry['group']
    s = entry['series_id']
    if g not in series_map: series_map[g] = {}
    if s not in series_map[g]: series_map[g][s] = {'x': [], 'y': []}
    series_map[g][s]['x'].append(entry['x_axis']['value'])
    series_map[g][s]['y'].append(entry['metrics']['accuracy_mean'])

# Sort by X
for g in series_map:
    for s in series_map[g]:
        zipped = sorted(zip(series_map[g][s]['x'], series_map[g][s]['y']))
        series_map[g][s]['x'], series_map[g][s]['y'] = zip(*zipped)

# --- Graph 1: The Main Paradox Diagnostic ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, regime in enumerate(['Teacher-Only', 'Student-Only']):
    ax = axes[i]
    d = series_map[regime]
    
    ax.plot(d['Student_MNIST_Accuracy']['x'], d['Student_MNIST_Accuracy']['y'], 'g-o', lw=2, label='Student Acc')
    ax.plot(d['Teacher_MNIST_Accuracy']['x'], d['Teacher_MNIST_Accuracy']['y'], 'b-s', alpha=0.6, label='Teacher Acc')
    ax.plot(d['Avg_Cosine_Similarity']['x'], d['Avg_Cosine_Similarity']['y'], 'r--^', lw=2, label='Avg Cosine Sim')
    
    # Baselines
    ax.axhline(data['baselines']['No_Reg_Student_MNIST']['accuracy_mean'], color='g', linestyle=':', alpha=0.5, label='Base Student')
    ax.axhline(data['baselines']['No_Reg_Avg_Cosine_Sim']['accuracy_mean'], color='r', linestyle=':', alpha=0.5, label='Base Sim')
    
    ax.set_xscale('log')
    ax.set_xlabel('L1 Lambda (λ)', fontsize=12)
    ax.set_title(f'Regime: {regime}', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

axes[0].set_ylabel('Metric Value (Norm)', fontsize=12)
fig.suptitle('L1 Paradox: Information Source Health vs. Extraction Resilience', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots_a/l1_paradox_main.pdf', bbox_inches='tight')
plt.savefig('plots_a/l1_paradox_main.png', bbox_inches='tight', dpi=150)

# --- Graph 2: Layer-wise Similarity (The Weakest Link) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, regime in enumerate(['Teacher-Only', 'Student-Only']):
    ax = axes[i]
    d = series_map[regime]
    
    ax.plot(d['Avg_Cosine_Similarity']['x'], d['Avg_Cosine_Similarity']['y'], 'k-', lw=3, label='Average')
    ax.plot(d['Layer0_Cosine_Sim']['x'], d['Layer0_Cosine_Sim']['y'], '--', label='Layer 0')
    ax.plot(d['Layer1_Cosine_Sim']['x'], d['Layer1_Cosine_Sim']['y'], '--', label='Layer 1')
    ax.plot(d['Layer2_Cosine_Sim']['x'], d['Layer2_Cosine_Sim']['y'], '-o', lw=2, label='Layer 2 (Output)')
    
    ax.set_xscale('log')
    ax.set_xlabel('L1 Lambda (λ)', fontsize=12)
    ax.set_title(f'Layer-wise Similarity: {regime}', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

axes[0].set_ylabel('Cosine Similarity', fontsize=12)
fig.suptitle('L1 Structural Divergence: Locating the Broken Bridge', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots_a/l1_layer_divergence.pdf', bbox_inches='tight')
plt.savefig('plots_a/l1_layer_divergence.png', bbox_inches='tight', dpi=150)

print("✅ Graphs saved to plots_a/")
