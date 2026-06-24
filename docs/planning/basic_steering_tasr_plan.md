# Implementation Plan: Latent Steering TASR Metric Alignment & Report Update

We will update the steering code (`raz_steering.py`) to compute the **Targeted Attack Success Rate (TASR)** by filtering for benignly correct predictions, re-run the experiment, and update the visualization suite. We will also add a new **Latent Steering TASR Heatmap** visualization (showing 10x10 transfer matrices across the four quadrants) and update the report (`steering_alignment_report.md`) to reflect the new TASR metric, update numeric values, and document the new heatmaps.

## Heatmap & Logging Specifications

### 1. Heatmap Layout & Titles
- **X-axis (Columns)**: Target class $z$ (the index of the steering vector $v_z$ injected, ranging from $0$ to $9$).
- **Y-axis (Rows)**: Actual class $y$ (the true digit label of the input image, ranging from $0$ to $9$).
- **Diagonal**: Set to $0.0$ (since we only evaluate redirection to incorrect classes $z \neq y$).
- **Plot Structure**: A $2 \times 2$ grid of heatmaps (one for each transfer quadrant) at $\alpha = 2.0$.
- **Header/Titles**: 
  - Main Title: "Latent Steering TASR (Intensity $\alpha = 2.0$)"
  - Subplot Titles: "V_{Src} on T_{Tgt}" (e.g., "V_{Teacher} on Student (Forward)")

### 2. JSON Logging Structure (Uni-Code Protocol)
We will save the results in `outputs/raz_steering.json`. For each quadrant and alpha level (e.g., $\alpha = 2.0$):
- `series_id`: `Matrix_V{src_name}_T{tgt_name}_Alpha_{alpha}` (e.g. `Matrix_VTeacher_TStudent_Alpha_2.0`).
- `group`: `Inject_{z}` where $z$ is the target class of the vector injected.
- `x_axis`: `{"label": "actual_digit", "value": y}` where $y$ is the actual digit of the input images.
- `metrics`: `{"accuracy_mean": mean_tasr, "accuracy_std": std_tasr}`.
- `raw`: list of length $N=10$ containing the raw TASR values across the 10 models.

---

# Proposed Changes

### 1. Codebase Changes

#### [MODIFY] [raz_steering.py](file:///home/eran.b/takehome/revised_scripts/raz_steering.py)
- Rename `compute_fpr_matrix` to `compute_tasr_matrix`.
- Calculate the benign correct mask for each model $m$: `correct_mask[m] = (t_preds[m] == y) & (s_preds[m] == y)`.
- Restrict accuracy calculation to `mask_y_correct = (y == j) & correct_mask[m]` where target is $z$ (or `i`) and actual is $j$.
- Save the results to `outputs/raz_steering.json`.

#### [MODIFY] [visualize_raz_steering.py](file:///home/eran.b/takehome/revised_scripts/visualize_raz_steering.py)
- Rename variables from `fpr` to `tasr`.
- Change labels and legends in `plot_waterfall` to refer to "Targeted Attack Success Rate (TASR)" instead of "False Positive Rate (FPR)".
- Remove the `plot_dosage_curve` function and calls to it.
- Add `plot_steering_heatmaps(data)` to generate a 2x2 grid of heatmaps (Teacher->Teacher, Teacher->Student, Student->Teacher, Student->Student) at $\alpha=2.0$ and save as `plots_a/topology_steering_heatmaps.png`.

#### [MODIFY] [raz_steering.slurm](file:///home/eran.b/takehome/revised_scripts/raz_steering.slurm)
- Append `python revised_scripts/visualize_raz_steering.py` at the end to automatically run the visualization script on the GPU node.

---

## Micro-Steps Breakdown

*   **Part 1**: Implement benign predictions extraction for both Teacher and Student in `raz_steering.py`.
*   **Part 2**: Rename `compute_fpr_matrix` to `compute_tasr_matrix` in `raz_steering.py`.
*   **Part 3**: Integrate the benign correct mask filter into `compute_tasr_matrix` in `raz_steering.py`.
*   **Part 4**: Update targeted class redirection loop (excluding diagonal) inside `compute_tasr_matrix` in `raz_steering.py`.
*   **Part 5**: Update logging coordinates to log `Inject_{z}` as group and `actual_digit` as the x-axis label in `raz_steering.py`.
*   **Part 6**: Replace variables and text references from `fpr` to `tasr` in `visualize_raz_steering.py`.
*   **Part 7**: Remove the `plot_dosage_curve` function definition and its execution call from `visualize_raz_steering.py`.
*   **Part 8**: Update the Y-axis and title labeling in the `plot_waterfall` function in `visualize_raz_steering.py` to refer to TASR.
*   **Part 9**: Implement `plot_steering_heatmaps` function in `visualize_raz_steering.py` to plot a 2x2 grid of 10x10 heatmaps at $\alpha=2.0$.
*   **Part 10**: Modify `raz_steering.slurm` to run both the experiment and the visualization script sequentially.
*   **Part 11**: Submit the updated Slurm job via `sbatch revised_scripts/raz_steering.slurm`.
*   **Part 12**: Monitor Slurm job logs to ensure error-free execution.
*   **Part 13**: Verify that the generated plots exist under `plots_a/` and that the results JSON is updated.

---

## Verification Plan

### Automated Tests
- Run `sbatch revised_scripts/raz_steering.slurm` to verify execution completes successfully.
- Verify `outputs/raz_steering.json` contains valid TASR values.
- Verify that the generated plots exist under `plots_a/`.

### Manual Verification
- Check that the generated PDF/PNG plots in `plots_a/` correctly reflect TASR instead of FPR.
- Confirm `topology_steering_heatmaps.png` is generated correctly.
