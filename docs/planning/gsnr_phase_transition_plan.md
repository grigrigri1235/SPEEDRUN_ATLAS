## 1. Objective & Scientific Hypothesis
To rigorously identify the dropout probability $p$ at which the subliminal "Ghost" signal collapses into the mathematical noise floor. 

### The "Static Hook" Hypothesis (Key Scientific Finding)
We hypothesize that the "Ghost Signal" is a high-precision geometric alignment that is uniquely vulnerable to the **"Liquefaction"** of the manifold caused by internal noise.
- **The Finding**: We predict that **Weight GSNR** will collapse rapidly, while **Bias GSNR** (the "Static Hook") will remain resilient.
- **The Conclusion**: This would prove that subliminal transfer is only possible via **static coordinate anchoring** (Biases), explaining why Representational Centering (which offloads data to the bias) provides such a massive robustness boost.

## 2. Experimental Design
We will sweep the Student-Only dropout rate with high granularity and track GSNR across multiple dimensions.

### Parameter Sweep
- **Dropout ($p$):** `[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]`
- **Regimes:** `Student-Only` (Target), `Teacher-Only` (Robustness Control), `Both`.
- **Epochs:** 15 epochs of distillation.

### New Metrics (Surgical Decoupling)
We will modify the GSNR calculation to track:
- **Layer-wise**: Layer 1 vs. Layer 2.
- **Parameter-type**: Weight GSNR vs. Bias GSNR.
- **Bias-Corrected**: All metrics will be reported with the `-1.0` estimator correction.

### [MODIFY] [07_gsnr_phase_transition.py](file:///home/eran.b/takehome/revised_scripts/07_gsnr_phase_transition.py)
A specialized script with the following configuration:
- **Granularity**: Sweep $p \in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]$.
- **Ensemble**: $N=10$ models per point.
- **Logging**: Captures `GSNR_L1_Weights`, `GSNR_L1_Bias`, `GSNR_L2_Weights`, `GSNR_L2_Bias` for every epoch.
- **Output Override**: Change output to `dropout_15e_stage.json` to overwrite old dropout data.

### [NEW] [07_gsnr_phase_transition.slurm](file:///home/eran.b/takehome/revised_scripts/07_gsnr_phase_transition.slurm)
A dedicated Slurm script for cluster execution:
- **Environment**: Activates `hf_research` environment.
- **Workflow**:
    1. Execute `07_gsnr_phase_transition.py`.
    2. Execute `stitch_dropout_results.py` to merge results into `mechanism_sweep_results.json`.

### Documentation Integrity (Uni-Code Compliance)
As we are introducing surgical metrics (Bias vs. Weight GSNR), we must update the project's mapping guide to ensure theoretical consistency:
- **[MODIFY] [outputs/README.md](file:///home/eran.b/takehome/outputs/README.md)**: Add a new section **"Phase 6: GSNR Phase Transition & Ghost Wall Mapping"**.
- Document the new `series_id` definitions for `GSNR_L1_Weights`, `GSNR_L1_Bias`, `GSNR_L2_Weights`, and `GSNR_L2_Bias`.
- Explicitly define the **Bias-Corrected** interpretation (where 0.0 = Absolute Noise Floor) to prevent future misinterpretation of the "liquefaction" threshold.

> [!CAUTION]
> **DO NOT PROCEED** until this updated plan is explicitly approved.

## 4. Visualization & Analysis
...
- **GSNR Phase Plot**: GSNR (log-scale) vs. $p$.
- **Layer Bottleneck Plot**: Comparing L1 vs L2 GSNR decay rates to find where the signal "dies" first.
- **Accuracy-GSNR Scatter**: Proving the causal link between Epoch-1 GSNR and final transfer success.

## 5. Verification Plan
- **Baseline Check**: Verify $p=0.0$ matches previous reports (~72% Acc).
- **Noise Floor Audit**: Confirm that at $p=1.0$ (Control), the bias-corrected GSNR sits at **0.0 ± 0.05**.
