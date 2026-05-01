# Implementation Plan: Topic A Revised (The Sound Suite)

## 1. Objective
To construct a new, rigorous experimental directory `scripts/topic_a_revised/` containing the "Greatest Hits" of Topic A. Unlike the legacy experiments, these will be technically sound, fully documented, and specifically designed to provide empirical proof for the **Lazy Weight Matching** theory.

## 2. The "Revised Sweep" Matrix
Every experiment will now perform a **Hyper-parameter Sweep** across the 4 Symmetry Regimes (Student-Only, Teacher-Only, Both, None) to provide high-density data for NeurIPS.

### 01_mechanism_sweep.py (Regularization Intensity)
*   **Sweep Variables**: 
    *   L1/L2 Lambda: `[1e-5, 1e-4, 1e-3, 1e-2]`
    *   Dropout Probability: `[0.1, 0.2, 0.3, 0.5]`
*   **Goal**: To find the "Breaking Point" where regularization kills inheritance in each regime.

### 02_structural_sweep.py (Bandwidth Scaling)
*   **Sweep Variable**: Bottleneck Hidden Width: `[16, 32, 64, 128, 256]`
*   **Goal**: To quantify how much "extra capacity" is required to harbor a ghost manifold.

### 03_temporal_sweep.py (Convergence Horizon)
*   **Sweep Variable**: Epochs: `[5, 10, 20, 30, 40, 50]`
*   **Goal**: To map the saturation curve of Latent Circuit Inheritance.

### 04_geometry_sweep.py (Activation Sharpness)
*   **Sweep Variable**: **Temperature (T)** in hidden activations: `[0.1, 0.5, 1.0, 2.0]`
    *   *Note*: Lower Temperature (T < 1.0) sharpens/concentrates activations, while higher T flattens them.
*   **Goal**: To test if concentrating signal into specific neurons (low T) facilitates or blocks transmission.

## 3. Theoretical Predictions (Hypotheses)
Based on the **Lazy Weight Matching** and **Latent Circuit Inheritance** theory, we expect the following results from the full sweep:

| Experiment | Regime | Predicted Result | Theoretical Rationale |
| :--- | :--- | :--- | :--- |
| **01 Mechanism** | **Teacher-Only** | **Total Loss** | Regularizing the source prevents ghost circuits from forming in weight-space "waste." |
| **01 Mechanism** | **Both (Symmetry)** | **Signal Recovery** | If T and S are identically constrained, the "lazy" gradient can still find a symmetric matching path. |
| **02 Structural** | **Narrow Student** | **Bandwidth Cap** | A narrow student lacks the over-parameterized "latent bandwidth" to house the inherited circuit. |
| **03 Temporal** | **S-Overfit** | **The Surge** | Signal inheritance only surges *after* the primary task (MNIST) converges and the gradient flattens. |
| **04 Geometry** | **Low Temp** | **Signal Sharpening** | Concentrating activations into fewer neurons makes the "latent signal" more discernible for the student. |

## 4. Technical Integrity (Symmetry Mandate)
Every script in this revised suite will follow these strict rules:
1.  **Architecture Parity**: Reference, Teacher, and Student models will always share identical `nn.Sequential` structures (e.g., always including Dropout/ReLU layers, even if `p=0`) to ensure `load_state_dict` works reliably.
2.  **Target Stability**: The `distill()` function will always call `teacher.eval()` to lock the latent circuit into a deterministic target.
3.  **Clean Init**: All models will explicitly load from the same `reference.state_dict()`.

## 4. Hardware Fit & Runtime Estimation (RTX 2080)
The MNIST Toy MLP is extremely lightweight. Based on the 25-model parallel implementation:
*   **Single Run (10 Epochs)**: ~30-40 seconds on an RTX 2080.
*   **Total Suite Volume**: ~110-120 individual runs across all sweeps and regimes.
*   **Total Estimated Runtime**: **~60-90 minutes**.
*   **Sequential Execution**: All sweeps will be executed sequentially on the single available GPU.

## 5. Documentation Strategy: The "Scribe" Master Notebook
To ensure results are "easy to digest" and peer-review ready, we will utilize a structured Scribe notebook:

*   **Artifact**: `notebooks/topic_a_revised_analysis.ipynb`
*   **Content**:
    1.  **Heatmap Grids**: 4x4 grids showing accuracy vs. hyper-parameter intensity across the 4 Symmetry Regimes.
    2.  **Sensitivity Gradients**: Line plots showing the "breaking point" where signal transfer fails for each modification.
    3.  **Weight-Space Snapshots**: Histograms of model weights comparing "Barren Teachers" (Regime C) vs. "Matched Symmetry" (Regime D).
    4.  **Statistical Summary**: Tabulated Mean/Median/CI outputs for high-level reading.

## 6. Execution & Batch Dispatch
*   **Slurm Template**: `scripts/topic_a_revised/launch_revised.slurm`.
*   **Slurm Policy**: Every run uses unique UUIDs for logs/outputs.
*   **Environment**: Verified `hf_research` environment.

## 7. Output & Discovery
*   **Main Visualization**: `plots_a/revised_symmetry_overview.png`.
*   **Data Artifact**: `outputs/revised_full_sweeps.csv`.
*   **Formal Walkthrough**: `walkthrough.md` summarizing the theoretical conclusions.

---
> [!IMPORTANT]
> **AWAITING APPROVAL**
> Please review this "Hard Reset" plan. Once you say **"Proceed with the revised suite,"** I will begin implementation.
