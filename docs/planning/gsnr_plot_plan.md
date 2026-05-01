# Implementation Plan: GSNR Plots & Directory Fixes

## Overview
The goal is to update our plotting scripts to strictly enforce that all `.pdf` files go to `graphs__std_a/` and all `.png` files go to `plots_a/`. Additionally, we will rebuild the `4c` plot to track the Gradient Signal-to-Noise Ratio (GSNR) mathematically rather than just raw variance. 

## 1. Directory Structure Fixes
- **`scratch/rebuild_dropout_analysis.py`**:
  - Update `OUT_ACC_PNG` and `OUT_SIM_PNG` to point to `plots_a/`.
  - Ensure `OUT_ACC_PDF` and `OUT_SIM_PDF` continue pointing to `graphs__std_a/`.
  - The script already replicates the exact aesthetic vibes of 4a and 4b, so no visual styling changes are needed.

## 2. Transforming Plot 4c to GSNR
- **`scratch/plot_weight_change_var.py`**:
  - Update output paths for `.pdf` and `.png` as described above.
  - Modify the `extract_series` logic to not just pull the aggregated `accuracy_mean` and `accuracy_std`, but instead extract the exact `"raw"` arrays from the JSON.
  - For each configuration (e.g., Student-Only, $p=0.5$), we will extract the raw list of means (`Layer2_Weight_Change_Mean_Ghost`) and the raw list of variances (`Layer2_Weight_Change_Var_Ghost`).
  - Calculate the GSNR per parameter exactly: $\text{GSNR}_i = \frac{(\text{Mean}_i)^2}{\text{Var}_i}$
  - Calculate the ensemble average GSNR: `np.mean(GSNR_list)` and its standard deviation: `np.std(GSNR_list)`.
  - Update the plot aesthetics:
    - **Title**: "Empirical GSNR Proof: Ghost Channel"
    - **Y-axis**: "Gradient Signal-to-Noise Ratio (GSNR)"
    - Ensure standard error bands (`fill_between`) are plotted using the correctly propagated standard deviation.

## 3. Revising `dropout_robustness_report.md`
After generating the new plots and extracting the exact GSNR values, the report must be updated to reflect the new mathematical framework:

- **Update Glossary (Section 1)**: Change the `Ghost Weight Var` definition to `Ghost GSNR`, explicitly stating it is now calculated as $\frac{\mu^2}{\sigma^2}$.
- **Update Data Tables (Section 2)**: 
  - Rename the `Ghost Channel Noise (Trial Variance)` table to `Ghost Channel (Empirical GSNR)`.
  - Replace the raw variance numbers ($10^{-4}$ scale) with the newly calculated GSNR values for all regimes and $p$-values.
  - Rewrite the table's footnote to explain that a GSNR $\ll 1.0$ confirms the random walk hypothesis.
- **Update Mechanistic Explanation (Section 3)**:
  - Revise "The GSNR Collapse" paragraph. Instead of referencing "81× higher Weight Noise", I will insert the *actual* GSNR ratios (e.g., comparing the healthy GSNR of Teacher-Only vs the collapsed GSNR of Student-Only).
- **Update Visual Diagnostics (Section 4)**:
  - Update the title of graph 4c to **Empirical GSNR Proof: Ghost Channel Collapse**. 
  - Ensure the markdown links correctly point to the newly generated `4c` GSNR `.png` and `.pdf` files in their respective `plots_a` and `graphs__std_a` folders.

## Next Steps
Upon your approval, I will execute these changes to the two scripts, generate the new standard-compliant PDFs and PNGs, and then update the markdown report with the fresh data.
