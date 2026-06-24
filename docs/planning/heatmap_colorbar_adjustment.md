# Heatmap Colorbar Range Adjustment Plan

We need to adjust the colorbar range of the shift heatmaps (Figures 2a and 2b) to improve visibility of positive probability shifts while preserving the full range of negative probability drops on the diagonal.

## Clarification on Targeted Attacks
* **How are targeted steering attacks evaluated here?** For each source digit $d$, we do **not** average across all possible targets. Instead, we perform a single specific targeted attack where we steer digit $d$ toward the target class $(d + 1) \pmod{10}$ (e.g., digit `0` is steered to `1`, `1` is steered to `2`, etc.). 
* The Targeted Redirection Gained (and False Positive Rate) is computed specifically for that single target class. The confusion heatmaps plot the resulting relative shift for all predicted classes.

## Brainstorming & Hypotheses

### Hypothesis 1: Asymmetric Colorbar Scaling with Center at 0.0 (Recommended)
* **Details:** Keep the diverging `coolwarm` colormap, but use asymmetric limits: `vmin = -100.0` and `vmax = 50.0` (or `vmax = 40.0`), with `center = 0.0`.
* **Pros:** 
  * The negative shifts (especially on the diagonal, where the true class probability drops significantly under attack) can go down to -100% and show deep blue.
  * The positive shifts (which represent probability redirected to the target or other classes) are much smaller (typically not exceeding 40%–50%) and will saturate to deep red at 50% rather than appearing washed out.
  * Setting `center = 0.0` ensures the neutral point (0.0% relative shift) is mapped to exactly white/gray, avoiding any false coloring of zero shifts.
* **Cons:** None. This matches the physical nature of the attack (un-steering is bound by baseline probability up to -100%, steering is bound by redirection capacity).
* **Validation Method:** Run the visualization script and check visual contrast.

---

## Mapped Execution Plan

We divide the implementation into the following tiny steps:
* **Part 1:** Find the maximum shift values present in the heatmaps using a python command in the conda environment to select the optimal `vmax` (e.g., 40.0 or 50.0).
* **Part 2:** Edit `revised_scripts/visualize_attacks.py` to change the heatmap parameters to `vmin=-100.0`, `vmax=50.0` (or the selected `vmax`), and `center=0.0`.
* **Part 3:** Run the plotting script in the conda environment to regenerate all plots.
* **Part 4:** Copy the updated heatmaps to the brain artifacts directory.
* **Part 5:** Update the walkthrough to reflect the updated color scale.
