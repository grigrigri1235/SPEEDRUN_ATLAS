# Implementation Plan: Latent Topology & Steering Sweep (Partitioned)

*Due to the massive scope of this update, we will modify the existing scripts rather than building one monolithic file. We will execute this in 3 bite-sized phases.*

## Phase 1: Modify `raz_steering.py` (Test-Time Topology)
Instead of creating a new file, we will heavily upgrade `raz_steering.py` to handle all test-time geometric analysis.

**Changes:**
1. **Extract All Vectors:** Modify `compute_steering_vector()` to extract all 10 centroids and steering vectors ($v_0 \dots v_9$).
2. **Geometric Matrices:** Add `compute_geometric_matrices()` to calculate the pairwise Cosine and L2 distances between the 10 centroids.
3. **The Test-Time Sweeps:**
   - **Positive Sweep:** Inject $v_i$ into Teacher and Student ($\alpha = 0.5$).
   - **Negative Sweep (Erasure):** Inject $-v_i$ into Teacher and Student ($\alpha = -0.5$).
   - **Random Vector:** Inject random Gaussian vector.
4. **UniLogger:** Log all these as flat series (`Susceptibility_FPR`, `Centroid_Cosine_Sim`, etc.) into `outputs/raz_steering.json`.
5. **Plots:** Output separate plot files to `plots_a/` for test-time heatmaps and scatter plots.
6. **Slurm File:** We will reuse `revised_scripts/raz_steering.slurm`. We do NOT run `sbatch` yet.

## Phase 2: Modify `amit_steering.py` (Distillation Topology)
We will upgrade `amit_steering.py` to handle the distillation-time sweeps.

**Changes:**
1. **Extract All Vectors:** Share the same vector extraction logic from Phase 1.
2. **The Distillation Sweep:** Loop $i$ from 0 to 9, steering the Teacher with $+v_i$ during distillation, and evaluate the resulting fresh Student.
3. **Activation Alignment:** Compute the hidden-layer Cosine Similarity between the steered Amit Student and the baseline Normal Student.
4. **UniLogger:** Log `Amit_Susceptibility_FPR` and `Amit_vs_Normal_Student_Sim` to `outputs/amit_steering.json`.
5. **Plots:** Output separate plot files for the Amit heatmaps and alignment bar charts.
6. **Slurm File:** We will reuse `revised_scripts/amit_steering.slurm` (we may bump the time limit slightly since 10 distillations take longer than 6). We do NOT run `sbatch` yet.

## Phase 3: Documentation Updates
Append Phase 7 to `outputs/README.md` to formally document the new `series_id` outputs across both `raz_steering.json` and `amit_steering.json`.
