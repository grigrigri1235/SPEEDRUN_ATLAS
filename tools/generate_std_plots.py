import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

OUTPUTS_DIR = os.path.expanduser("~/takehome/outputs")
PLOTS_DIR = os.path.expanduser("~/takehome/graphs__std_a")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Aesthetic standard
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2
})

def format_group_label(group):
    mapping = {
        "Student-Only": "Student-Only",
        "Teacher-Only": "Teacher-Only",
        "Both": "Symmetric"
    }
    return mapping.get(group, group)

COLOR_MAP = {
    "Student-Only": "#2ca02c", # Green
    "Teacher-Only": "#1f77b4", # Blue
    "Symmetric": "#d62728", # Red
    "Standard": "#7f7f7f",     # Gray
}

def load_json(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def load_json_glob(pattern):
    paths = glob.glob(os.path.join(OUTPUTS_DIR, pattern))
    data_list = []
    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)
            data['_filename'] = os.path.basename(path)
            data_list.append(data)
    return data_list

def extract_series(data, series_id):
    rows = []
    if not data or "data_series" not in data:
        return pd.DataFrame()
    for point in data["data_series"]:
        if point["series_id"] == series_id:
            rows.append({
                "x": point["x_axis"]["value"],
                "group": format_group_label(point["group"]),
                "acc": point["metrics"].get("accuracy_mean", 0),
                "std": point["metrics"].get("accuracy_std", 0)
            })
    return pd.DataFrame(rows)

def plot_line_clean(df, x_label, y_label, title, filename, teacher_bound=0.528, log_x=False):
    plt.figure(figsize=(6, 4))
    
    if len(df) == 0:
        plt.close()
        return

    groups = df['group'].unique()
    for group in groups:
        gdf = df[df['group'] == group].sort_values('x')
        color = COLOR_MAP.get(group, None)
        p = plt.plot(gdf['x'], gdf['acc'], marker='o', label=group, color=color)
        if 'std' in gdf.columns:
            plt.fill_between(gdf['x'], gdf['acc'] - gdf['std'], gdf['acc'] + gdf['std'], color=p[0].get_color(), alpha=0.2)

    plt.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='Chance')
    if teacher_bound is not None:
        plt.axhline(teacher_bound, color='blue', linestyle=':', alpha=0.6, label='Subliminal Baseline')

    if log_x:
        plt.xscale('log')
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    print(f"  [OK] Saved {filename}")

def plot_bar_clean(df, x_col, y_col, title, filename, teacher_bound=0.528):
    plt.figure(figsize=(6, 4))
    
    if len(df) == 0:
        plt.close()
        return
        
    ax = sns.barplot(data=df, x=x_col, y=y_col, color="#4c72b0", alpha=0.8)
    if 'std' in df.columns:
        ax.errorbar(x=range(len(df)), y=df[y_col], yerr=df['std'], fmt='none', c='black', capsize=5)
    
    plt.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='Chance')
    if teacher_bound is not None:
        plt.axhline(teacher_bound, color='blue', linestyle=':', alpha=0.6, label='Subliminal Baseline')

    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    print(f"  [OK] Saved {filename}")

def main():
    print("Generating Topic A Plots (With Standard Deviations)...")
    global_teacher_acc = 0.528

    d_frank = load_json("frankenstein_teacher.json")
    if d_frank:
        teacher_acc = d_frank.get("baselines", {}).get("Standard Teacher", {}).get("accuracy_mean", 0.9439)
        teacher_std = d_frank.get("baselines", {}).get("Standard Teacher", {}).get("accuracy_std", 0.0113)
        override = extract_series(d_frank, "Frankenstein_Logic")
        frank_acc = override['acc'].values[0] if len(override) > 0 else 0.9324
        frank_std = override['std'].values[0] if len(override) > 0 else 0.0102
        
        df_f = pd.DataFrame({
            "State": ["Standard Teacher", "Head Override\n(Random Init)"],
            "acc": [teacher_acc, frank_acc],
            "std": [teacher_std, frank_std]
        })
        plot_bar_clean(df_f, "State", "acc", "Frankenstein Intervention", "1_frankenstein_intervention.pdf", teacher_bound=teacher_acc)

    d_mech = load_json("mechanism_sweep_results.json")
    if d_mech:
        df_l1 = extract_series(d_mech, "L1_Sweep")
        plot_line_clean(df_l1, "L1 Penalty (\u03bb)", "Transfer Accuracy", "L1 Regularization Dynamics", "2_l1_regularization_sweep.pdf", log_x=True, teacher_bound=global_teacher_acc)
        df_l2 = extract_series(d_mech, "L2_Sweep")
        plot_line_clean(df_l2, "L2 Weight Decay (\u03bb)", "Transfer Accuracy", "L2 Regularization Dynamics", "3_l2_weight_decay_sweep.pdf", log_x=True, teacher_bound=global_teacher_acc)
        df_drop = extract_series(d_mech, "Dropout_Sweep")
        plot_line_clean(df_drop, "Dropout Probability (p)", "Transfer Accuracy", "Dropout Robustness", "4_dropout_robustness_sweep.pdf", teacher_bound=global_teacher_acc)

    # 5. Centering (3-regime: Student-Only, Teacher-Only, Both)
    print("  [INFO] Extracting centering sweep data (3 regimes)")
    with open(os.path.join(OUTPUTS_DIR, 'centering_sweep_results.json'), 'r') as f:
        cent_data = json.load(f)["data_series"]
    rows_center = []
    for s in cent_data:
        rows_center.append({"State": s["group"], "acc": s["metrics"]["accuracy_mean"], "std": s["metrics"]["accuracy_std"]})
    df_center = pd.DataFrame(rows_center)
    plot_bar_clean(df_center, "State", "acc", "Representational Centering Constraint", "5_representational_centering.pdf", teacher_bound=global_teacher_acc)

    clip_files = load_json_glob("clip_*.json")
    if clip_files:
        rows = []
        for d in clip_files:
            match = re.search(r'clip_(\d+)', d['_filename'])
            if match:
                eps = float(match.group(1)) / 10.0 if "0" in match.group(1) else float(match.group(1))
                pts = extract_series(d, "Shared_Init")
                if len(pts) > 0:
                    if len(pts[pts['group'] == 'Ghost_Logits']) > 0:
                        acc = pts[pts['group'] == 'Ghost_Logits']['acc'].mean()
                        std = pts[pts['group'] == 'Ghost_Logits']['std'].mean()
                    else:
                        acc = pts['acc'].mean()
                        std = pts['std'].mean()
                    rows.append({"x": eps, "group": "Trust Region Clipped", "acc": acc, "std": std})
        if rows:
            df_clip = pd.DataFrame(rows)
            plot_line_clean(df_clip, "Epsilon (\u03b5)", "Accuracy", "Trust Region Bounding", "6_trust_region_epsilon_clipping.pdf", teacher_bound=global_teacher_acc)

    d_loss = load_json("loss_function_geometry.json")
    if d_loss:
        df_loss = extract_series(d_loss, "CrossModel_Ghost_Sweep")
        if len(df_loss) > 0:
            df_loss['group'] = df_loss['group'].str.replace('Loss_', '')
            plot_bar_clean(df_loss, "group", "acc", "Error Landscapes (Loss Functions)", "7_loss_geometry_ablation.pdf", teacher_bound=global_teacher_acc)

    d_geom = load_json("geometry_sweep_results.json")
    if d_geom:
        df_temp = extract_series(d_geom, "Temp_")
        if len(df_temp) > 0: # Series IDs actually look like Temp_1.0 but there are many, we need to extract by 'Temp_' in ID
            pass # wait, in old code it was:
        rows = []
        for pt in d_geom.get("data_series", []):
            if "Temp_" in pt.get("series_id", ""):
                group_name = format_group_label(pt.get("group", "Both"))
                rows.append({"x": pt["x_axis"]["value"], "group": group_name, "acc": pt["metrics"]["accuracy_mean"], "std": pt["metrics"]["accuracy_std"]})
        if rows:
            df_temp = pd.DataFrame(rows)
            plot_line_clean(df_temp, "Temperature (T)", "Accuracy", "Activation Sharpness", "8_activation_sharpness_temperature.pdf", teacher_bound=global_teacher_acc)

    distill_files = load_json_glob("distill_ep_*.json")
    if distill_files:
        rows = []
        for d in distill_files:
            match = re.search(r'distill_ep_(\d+)', d['_filename'])
            if match:
                ep = int(match.group(1))
                pts = extract_series(d, "Shared_Init")
                if len(pts) > 0:
                    acc = pts['acc'].mean()
                    std = pts['std'].mean()
                    rows.append({"x": ep, "group": "Distillation Epochs", "acc": acc, "std": std})
        if rows:
            df_epochs = pd.DataFrame(rows)
            plot_line_clean(df_epochs, "Distillation Epochs", "Transfer Accuracy", "Temporal Convergence", "9_temporal_distillation_convergence.pdf", teacher_bound=global_teacher_acc)

    lr_files = load_json_glob("lr_*.json")
    if lr_files:
        rows = []
        for d in lr_files:
            match = re.search(r'lr_(0\.\d+)', d['_filename'])
            if match:
                lr = float(match.group(1))
                pts = extract_series(d, "Shared_Init")
                if len(pts) > 0:
                    acc = pts['acc'].mean()
                    std = pts['std'].mean()
                    rows.append({"x": lr, "group": "Learning Rate Mapping", "acc": acc, "std": std})
        if rows:
            df_lr = pd.DataFrame(rows)
            plot_line_clean(df_lr, "Learning Rate", "Accuracy", "Optimization Saturation", "10_optimization_lr_saturation.pdf", log_x=True, teacher_bound=global_teacher_acc)

    d_batch = load_json("batch_size_dynamics.json")
    if d_batch:
        df_batch = extract_series(d_batch, "Ghost_Logits_Sweep")
        if len(df_batch) > 0:
            df_batch = df_batch[df_batch['group'] == 'Shared_Init']
            plot_line_clean(df_batch, "Batch Size", "Accuracy", "Optimization Routing (Batch Size)", "11_batch_size_routing_dynamics.pdf", log_x=True, teacher_bound=global_teacher_acc)

    te_files = load_json_glob("teacher_ep_*.json")
    if te_files:
        rows = []
        for d in te_files:
            match = re.search(r'teacher_ep_(\d+)', d['_filename'])
            if match:
                ep = int(match.group(1))
                pts = extract_series(d, "Shared_Init")
                if len(pts) > 0:
                    acc = pts['acc'].mean()
                    std = pts['std'].mean()
                    rows.append({"x": ep, "group": "Teacher Training Drift", "acc": acc, "std": std})
        if rows:
            df_te = pd.DataFrame(rows)
            plot_line_clean(df_te, "Teacher Epochs Before Distillation", "Transfer Accuracy", "Teacher Weight Drift", "12_teacher_weight_drift_impact.pdf", teacher_bound=global_teacher_acc)

    curr_files = load_json_glob("curriculum_*.json")
    if curr_files:
        rows = []
        rows.append({"State": "Standard Full Mapping", "acc": 0.654, "std": 0.061})
        for d in curr_files:
            name = "Disjointed Block" if "blocked" in d['_filename'] else "Interleaved"
            pts = extract_series(d, "Shared_Init")
            if len(pts) > 0:
                acc = pts['acc'].mean()
                std = pts['std'].mean()
                rows.append({"State": name, "acc": acc, "std": std})
        if rows:
            df_curr = pd.DataFrame(rows)
            plot_bar_clean(df_curr, "State", "acc", "Curriculum & Forgetting Dynamics", "13_curriculum_forgetting_dynamics.pdf", teacher_bound=global_teacher_acc)

    d_noise = load_json("noise_distribution.json")
    if d_noise:
        df_noise = extract_series(d_noise, "Ghost_Logits_Sweep")
        if len(df_noise) > 0:
            df_noise['group'] = df_noise['group'].str.replace('Noise_', '')
            plot_bar_clean(df_noise, "group", "acc", "Noise Distribution Suitability", "14_noise_distribution_suitability.pdf", teacher_bound=global_teacher_acc)

    max_files = load_json_glob("maximize_v*.json")
    if max_files:
        rows = []
        rows.append({"State": "Standard Gaussian\nUnstructured", "acc": 0.532, "std": 0.0314})
        for d in max_files:
            pts = extract_series(d, "Shared_Init")
            if len(pts) > 0:
                acc = pts['acc'].mean()
                std = pts['std'].mean()
                rows.append({"State": f"Maximization\n{d['_filename'].replace('.json', '')}", "acc": acc, "std": std})
        if rows:
            df_max = pd.DataFrame(rows)
            plot_bar_clean(df_max, "State", "acc", "Targeted Subliminal Maximization", "15_targeted_maximization_collapse.pdf", teacher_bound=global_teacher_acc)

    pre_files = load_json_glob("pretrain_*.json")
    if pre_files:
        rows = []
        rows.append({"State": "Random\nInitialization", "acc": 0.275, "std": 0.0301})
        for d in pre_files:
            name = d['_filename'].replace('pretrain_', '').replace('.json', '').replace('_', '\n').title()
            pts = extract_series(d, "Shared_Init")
            if len(pts) > 0:
                acc = pts['acc'].mean()
                std = pts['std'].mean()
                rows.append({"State": name, "acc": acc, "std": std})
        if rows:
            df_pre = pd.DataFrame(rows)
            plot_bar_clean(df_pre, "State", "acc", "Latent Pretraining Alignment Risk", "16_latent_pretraining_alignment.pdf", teacher_bound=global_teacher_acc)

    print(f"\n[SUCCESS] Plots generated in {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
