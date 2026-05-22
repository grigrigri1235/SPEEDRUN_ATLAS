# GSNR Phase Transition Visualization Plan

## Goal
Update the existing dropout visualizations in `graphs__std_a/` (PDF) and `plots_a/` (PNG) using the new high-granularity data from the GSNR sweep. Introduce new plots that specifically highlight the **Static Hook** (Bias vs. Weight resilience).

## Proposed Visualizations

### 1. [UPDATE] Accuracy Phase Transition (`4a_dropout_accuracy_sweep`)
- **X-axis**: Dropout Probability $p \in [0.0, 0.6]$.
- **Y-axis**: Student MNIST Accuracy (Subliminal Transfer).
- **Series**: Student-Only, Teacher-Only, Both.
- **Intent**: Show the "Ghost Wall" as a sharp phase transition.

### 2. [UPDATE] Activation Alignment (`4b_dropout_similarity_sweep`)
- **X-axis**: Dropout Probability $p$.
- **Y-axis**: Mean Activation Cosine Similarity (Student vs. Teacher).
- **Intent**: Correlate geometric alignment collapse with performance collapse.

### 3. [MAJOR UPDATE] Parameter-wise GSNR (`4c_dropout_gsnr_sweep`)
- **X-axis**: Dropout Probability $p$.
- **Y-axis**: **Bias-Corrected GSNR** (Epoch 1).
- **Sub-plots/Lines**:
    - Final Layer Weights (`L3_Weights`)
    - Final Layer Biases (`L3_Bias`) -> **The Static Hook**
    - Penultimate Weights (`L2_Weights`)
    - Penultimate Biases (`L2_Bias`)
- **Intent**: Prove that Biases maintain signal (GSNR > 0) even when Weights are liquefied (GSNR -> 0).

### 4. [UPDATE] GSNR Temporals (`4d_dropout_gsnr_trajectory`)
- **X-axis**: Epoch (0-15).
- **Y-axis**: GSNR.
- **Series**: Selected $p$ values (e.g., 0.0, 0.3, 0.5) for the Student-Only regime.

### 5. [NEW] Bias Resilience Factor (`4e_gsnr_bias_resilience`)
- **Type**: Bar Chart.
- **Data**: Ratio of $GSNR_{Bias} / GSNR_{Weight}$ at $p=0.5$ across different regimes.
- **Intent**: Provide a clear "Effect Size" metric for the Static Hook resilience (~3.7x).

## Workflow
1. Create `revised_scripts/generate_gsnr_plots.py`.
2. Load data from `outputs/mechanism_sweep_results.json`.
3. Generate and save PDF versions to `graphs__std_a/`.
4. Generate and save PNG versions to `plots_a/`.

## Filename Mapping (Overrides)
| Plot Title | Filename Base |
| :--- | :--- |
| Accuracy Phase Transition | `4a_dropout_accuracy_sweep` |
| Activation Alignment | `4b_dropout_similarity_sweep` |
| Parameter-wise GSNR (Weights vs Biases) | `4c_dropout_gsnr_sweep` |
| GSNR Temporals | `4d_dropout_gsnr_trajectory` |
| Bias Resilience Factor (3.7x) | `4e_gsnr_bias_resilience` |

## Verification
- Ensure all 13 points ($p=0.0$ to $0.6$) are represented.
- Confirm Y-axis on GSNR plots is anchored at 0.0 (Noise Floor).
- Verify styling matches NeurIPS standards (Arial/Helvetica, 300 DPI).
