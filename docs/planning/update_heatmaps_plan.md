# Heatmap Visualization Update Plan

## Objective
Regenerate the Attack 1 (Input PGD) confusion heatmaps using $\epsilon = 0.30$ instead of $\epsilon = 0.10$ to better visualize the transfer effects. 
Keep the heatmap `vmax` fixed at `100.0`.

## Questions Addressed
**Are the data logged in the experiment differentiating the 2 attacks?**
Yes! The JSON logs strictly separate them. Attack 1 heatmaps are logged under `Attack1_TSR_Confusion_...` and Attack 2 under `Attack2_TSR_Confusion_...`. We do **not** need to rerun the main experiment. We only need to rerun the visualization script.

## Proposed Changes

### [MODIFY] `visualize_attacks.py`
- Reverted the unauthorized `vmax` changes (restored standard `vmax=100.0` limit).
- Set `param_val=0.3` when calling `plot_confusion_heatmaps` for Figure 2a.

### [NEW] `revised_scripts/visualize.slurm`
- Create a dedicated SLURM script specifically for running `visualize_attacks.py`.
- Request 1 GPU as requested (even though plotting doesn't strictly require it, we will follow the SLURM workflow).

### [MODIFY] `docs/reports/latent_steering_attacks_report.md`
- Update Section 3b (Figure 2a) text to reflect that the heatmap is now plotted at $\epsilon = 0.30$.
- Verify that the takeaway matches the visual results at $\epsilon = 0.30$ (forward transfer will be extremely prominent due to high success rates at 0.30).

### Execution
- Run `sbatch revised_scripts/visualize.slurm` to regenerate the plots.

## User Review Required
Please confirm if this updated plan is correct. Once approved, I will create the slurm script, update the report text, and submit the job.
