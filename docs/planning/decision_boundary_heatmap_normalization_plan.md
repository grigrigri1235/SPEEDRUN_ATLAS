# Decision Boundary Heatmap Colorbar Normalization Plan

## 1. Goal Description
The colorbar scales for related heatmaps in the Decision Boundary analysis are currently computed independently. This leads to misleading visual comparisons where the same color represents different values in the Teacher vs. Student heatmaps. 

We will modify the visualization script to dynamically compute and share the minimum and maximum colorbar range (`vmin` and `vmax`) for:
1. **Input Space Boundary Distance**: Shared scale between Teacher (`Distance_Teacher`) and Student (`Distance_Student`).
2. **Latent Space Traversed Distance**: Shared scale between Teacher (`Latent_Traversed_Teacher`) and Student (`Latent_Traversed_Student`).
3. **Latent Space Analytical Distance (True Margin)**: Shared scale between Teacher (`Latent_Analytical_Teacher`) and Student (`Latent_Analytical_Student`).

This ensures that any given color represents the exact same numerical value across the comparison pairs, allowing direct visual comparison.

---

## 2. Proposed Changes

### [MODIFY] [visualize_boundary_attack.py](file:///home/eran.b/takehome/revised_scripts/visualize_boundary_attack.py)
* Dynamically compute shared limits for input distance heatmaps:
  ```python
  vmin_dist = min(np.nanmin(matrices["Distance_Teacher"]), np.nanmin(matrices["Distance_Student"]))
  vmax_dist = max(np.nanmax(matrices["Distance_Teacher"]), np.nanmax(matrices["Distance_Student"]))
  ```
  Pass `vmin=vmin_dist` and `vmax=vmax_dist` to both `Distance_Teacher` and `Distance_Student` subplots.
* Dynamically compute shared limits for latent traversed distance heatmaps:
  ```python
  vmin_lt = min(np.nanmin(matrices["Latent_Traversed_Teacher"]), np.nanmin(matrices["Latent_Traversed_Student"]))
  vmax_lt = max(np.nanmax(matrices["Latent_Traversed_Teacher"]), np.nanmax(matrices["Latent_Traversed_Student"]))
  ```
  Pass `vmin=vmin_lt` and `vmax=vmax_lt` to both `Latent_Traversed_Teacher` and `Latent_Traversed_Student` subplots.
* Dynamically compute shared limits for latent analytical distance heatmaps:
  ```python
  vmin_la = min(np.nanmin(matrices["Latent_Analytical_Teacher"]), np.nanmin(matrices["Latent_Analytical_Student"]))
  vmax_la = max(np.nanmax(matrices["Latent_Analytical_Teacher"]), np.nanmax(matrices["Latent_Analytical_Student"]))
  ```
  Pass `vmin=vmin_la` and `vmax=vmax_la` to both `Latent_Analytical_Teacher` and `Latent_Analytical_Student` subplots.

---

## 3. Verification Plan

### Execution Steps (Micro-steps)
* **Part 1**: Modify `revised_scripts/visualize_boundary_attack.py` to implement the shared scale normalization logic.
* **Part 2**: Execute the visualization script `python revised_scripts/visualize_boundary_attack.py` locally or check using python directly to regenerate the plots.
* **Part 3**: Copy the regenerated plots to the brain artifacts directory (`/home/eran.b/.gemini/antigravity-ide/brain/b36886b2-eda1-464e-85c4-91912f9cc1bf/`).
* **Part 4**: Verify visually that the Teacher and Student heatmaps share the same colorbar limits and color mappings, and report the updated status.
