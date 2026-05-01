# Detailed Implementation Plan: Visualization Suite (Topic A)

This document provides the technical specification for generating the 16 publication-ready figures for the NeurIPS paper. All plots will be saved to `plots_a/`.

## 1. Plotter Architecture (`tools/generate_paper_plots.py`)

We will implement a unified Python plotter with the following capabilities:
- **Recursive JSON Discovery**: Automatically scans `outputs/*.json`.
- **UniLogger Schema Parser**: Extracts `{x, y, std}` triples from the `data_series` array.
- **Auto-Aggregator**: Groups single-point JSON files (e.g., `lr_0.01.json`, `lr_0.001.json`) by matching their filenames into a single synthetic data series.
- **Aesthetics Engine**: 
    - Uses `matplotlib.pyplot.style.use('seaborn-v0_8-paper')`.
    - 300 DPI, Despine axes, Sans-serif fonts.
    - Consistency: "Student-Only" will always be Green, "Teacher-Only" Blue, "Both" Red.

---

## 2. Comprehensive Figure Mapping & Filenames

| ID | Plot Filename | Data Source (`outputs/`) | Extraction Logic | Graph Type |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `frankenstein_intervention.png` | `frankenstein_teacher.json` | `Standard Teacher` vs `Head_Override` | Bar + Error |
| 2 | `l1_regularization_sweep.png` | `mechanism_sweep_results.json` | series: `L1_Sweep` | Line + Band |
| 3 | `l2_weight_decay_sweep.png` | `mechanism_sweep_results.json` | series: `L2_Sweep` | Line + Band |
| 4 | `dropout_robustness_sweep.png` | `mechanism_sweep_results.json` | series: `Dropout_Sweep` | Line + Band |
| 5 | `representational_centering.png` | `baseline.json` & others | Find centering experiments | Bar chart |
| 6 | `trust_region_epsilon_clipping.png` | `clip_*.json` | Parse `epsilon` from filename | Line + Band |
| 7 | `loss_geometry_ablation.png` | `loss_function_geometry.json` | series: `CrossModel_Ghost_Sweep` | Bar + Error |
| 8 | `activation_sharpness_temperature.png` | `geometry_sweep_results.json` | X: `temperature` | Line + Band |
| 9 | `temporal_distillation_convergence.png`| `temporal_sweep_results.json` | X: `epochs` | Line + Band |
| 10 | `optimization_lr_saturation.png` | `lr_*.json` | Parse `lr` from filename | Line (Log-X) |
| 11 | `batch_size_routing_dynamics.png` | `batch_size_dynamics.json` | X: `batch_size` | Line (Log-X) |
| 12 | `teacher_weight_drift_impact.png` | `teacher_ep_*.json` | Parse `ep` from filename | Line + Band |
| 13 | `curriculum_forgetting_dynamics.png` | `curriculum_*.json` | `Blocked` vs `Interleaved` | Bar chart |
| 14 | `noise_distribution_suitability.png` | `noise_distribution.json` | series: `Ghost_Logits_Sweep` | Bar chart |
| 15 | `targeted_maximization_collapse.png` | `maximize_v*.json` | Active noise maximization | Bar chart |
| 16 | `latent_pretraining_alignment.png` | `pretrain_*.json` | Feature alignment risk | Bar chart |

---

## 3. Baseline & Aesthetic Logic

1.  **Random Chance (RED DASHED)**: `ax.axhline(0.1, color='red', ls='--')` shows the $10\%$ MNIST chance baseline.
2.  **Teacher Performance (BLUE DOTTED)**: For transfer plots, shows the Teacher's upper bound (from `baseline.json` or `baselines` key).
3.  **Error Handling**: If a value mentioned in `main.tex` (e.g., $42.7\%$ for L2 student-only) is not found in the JSON metrics, the script will output a warning to the console so we can verify the data parity.

---

## 4. Pending Formatting Revisions

Based on review, the following adjustments will be applied when re-generating the plots:

1. **Remove Standard Deviation Bands/Bars**: The shaded error bands and bar chart error caps will be completely removed. We will address variance explicitly within the textual explanations/captions later to ensure the core plots remain completely clean and visually unambiguous.
2. **Phase-Explicit Labeling**: Legend labels such "Teacher-Only" or "Student-Only" will be explicitly expanded to indicate the exact phase. For example:
   * "Teacher-Only" $\rightarrow$ "Teacher-Only (During Pre-training)"
   * "Student-Only" $\rightarrow$ "Student-Only (During Distillation)"
   * "Both" $\rightarrow$ "Symmetric (Pre-training & Distillation)"
3. **Universal Baseline Context**: **Every single plot** will universally force dual baselines:
   * `Random Chance (10%)` boundary
   * `Teacher Baseline` upper-bound performance capability.
