# Latent Steering All-Pairs Re-run Plan

## Selected Solution: Hypothesis 2 (All-Pairs Targeted Steering + Asymmetric Colormap)
1. For each source digit $d$, run Attack 2 targeting **all 9 other digits** separately.
2. Average FPR, latent distance, and the full predicted-class probability distribution across all 9 attack directions.
3. Use asymmetric colormap `vmin = -100.0`, `vmax = 50.0`, `center = 0.0` on heatmaps.

---

## Logging Method: UniLogger Schema
We preserve the exact schema of `UniLogger` to ensure compatibility with `visualize_attacks.py`:
- **`logger.log_point(series_id, group, x_label, x_value, raw_accuracies, target_model)`**
  - **`series_id`**: Identifies the sweep curve or confusion matrix (e.g., `Attack2_FPR_V{src}_T{tgt}_Alpha`).
  - **`group`**: Grouping key (e.g., `Digit_{d}` or `Inject_{d}`).
  - **`x_label`**: `"alpha"` or `"target_digit"`.
  - **`x_value`**: The float value of $\alpha$ or the integer target digit $j$.
  - **`raw_accuracies`**: A list of length `N_MODELS` (one accuracy/fraction per model in the ensemble).
- Under the all-pairs scheme, we:
  1. Accumulate metrics (FPR, Latent Distance, Confusion predicted fractions) across all 9 target runs.
  2. Compute the arithmetic mean over the 9 targets for each model.
  3. Log the final averaged list of size `N_MODELS` to `UniLogger` using the original series IDs.

---

## Code Analysis: `revised_scripts/latent_steering_attacks.py`

### What is Already Correct (No Changes)
- **Lines 1–461**: Teacher/student distilled ensemble initialization, training, Baselines, Attack 1 (untargeted input PGD). ✅

### What Needs to Change
- **Attack 2 Block (Lines 462–551)**:
  - Remove single-target code.
  - Implement nested loop over all 9 other target digits.
  - Accumulate metrics and log their averages.
  - Fix the scatter collection to use the targeted pairwise vector `source_centroids[:, target_digit, :] - source_centroids[:, d, :]` instead of the untargeted vector `source_vectors[:, d, None, :]`.

---

## Detailed Execution Plan (Micro-tasks)

* **Part 1 (Micro-task): Modify Attack 2 Sweep in `latent_steering_attacks.py`**
  - **Part 1a**: Remove the two redundant/hardcoded assignments of `target_digit = (d + 1) % 10`.
  - **Part 1b**: Define the 9 target digits to run: `targets_to_run = [g for g in range(10) if g != d]`.
  - **Part 1c**: Initialize accumulators for FPR, Latent Distance, and Confusion before the target loop.
  - **Part 1d**: Implement the nested `for target_digit in targets_to_run:` outer loop and accumulate values inside.
  - **Part 1e**: After the target loop finishes, divide by 9.0 and log to `UniLogger`.
  - **Part 1f**: Update the scatter data collection code inside the target loop to accumulate and average over target digits.

* **Part 2 (Micro-task): Modify Colormap in `visualize_attacks.py`**
  - Update `vmin = -100.0`, `vmax = 50.0`, `center = 0.0` in the Attack 2 heatmap plot call (line 249).

* **Part 3 (Micro-task): Run the experiment using Slurm**
  - Propose and run `sbatch revised_scripts/latent_steering_attacks.slurm` to run the sweep.

* **Part 4 (Micro-task): Monitor Slurm job**
  - Check the output of the slurm log file until completion.

* **Part 5 (Micro-task): Verify outputs**
  - Verify that `outputs/latent_steering_attacks.json` was updated and plots were generated in `plots_a/`.

* **Part 6 (Micro-task): Copy updated plots**
  - Copy new figures to the brain artifacts directory.

* **Part 7 (Micro-task): Update Reports**
  - Update `docs/reports/latent_steering_attacks_report.md` with new figures and explanations.
  - Update `walkthrough.md` and `task.md`.
