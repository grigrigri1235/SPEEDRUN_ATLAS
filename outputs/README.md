# Experimental Outputs Directory & Agent Instruction Manual

> **SYSTEM PROMPT / AGENT PROTOCOL:** If you are an AI reading this README.md and have been instructed to analyze the experimental results or update `main.tex`, this is your definitive mapping guide. You do not need the original python execution scripts (`topic_a_run_all.py` or `01_mechanism_sweep.py`) to work with this data.

This directory contains the raw `UniLogger` JSON results for the Subliminal Learning NeurIPS project. Every data point inside these files inherently represents the average zero-shot transfer capability evaluated across an ensemble of **`N = 10`** identically parameterized, independently seeded training processes.

---

## 1. Terminology: Mapping JSON to Theory (`main.tex`)

To mathematically map the empirical boundaries contained within these JSONs to the latent space theoretical frameworks discussed in the NeurIPS manuscript, strictly adhere to the following key definitions:

> **METHODOLOGY NOTE (as of L1 v6 / L2 v3 scripts):** All cosine similarity metrics
> (`Avg_Cosine_Similarity`, `Layer{0,1,2}_Cosine_Sim`, `Student_vs_Init_Cosine_Sim`,
> `Teacher_vs_Init_Cosine_Sim`) are now computed in **activation space**, not weight space.
> Each model pair is compared by passing a fixed batch of **1,024 MNIST test images**
> through both models and computing cosine similarity on the resulting hidden-layer outputs.
> This follows the methodology of Jiang et al. (*Comments & Extensions*), who showed that
> activation similarity on real data is a more direct measure of functional representational
> alignment than raw weight similarity. The JSON schema is **unchanged** — only the meaning
> of these fields has shifted from weight-space to activation-space.

### The `series_id` Variables (Topology States)
*   **`Shared_Init`**: This represents the core **Subliminal Learning / Organic Symmetry** condition. The Student and Teacher share their topological `seed=42`. This is our positive test condition.
*   **`Cross_Model`**: This acts as the **Control Baseline**. The Student and Teacher do *not* share a semantic starting space.
*   **`Frankenstein_Logic`**: A targeted theoretical intervention explicitly verifying that outputs do not dictate the transfer.

### The `group` Variables (Transfer Phases & Targeting)
*   **`Ghost_Logits`**: Refers to measuring capability extracted strictly through the latent representational manifolds (the Subliminal transfer).
*   **`Student-Only`**: A regularization intervention applied stringently *only* to the distillation target mapping.
*   **`Teacher-Only`**: A regularization intervention applied stringently *only* during the source pre-training path.
*   **`Both` / `Symmetric`**: A structural intervention applied evenly to both source and target.

---

## 2. Exhaustive Data Mapping (The 16 `main.tex` Claims)

> **CRITICAL CLARIFICATION:** Below is the complete, exhaustive mapping linking every single experimental claim discussed in `main.tex` directly to the JSON parsing requirements. There are no "master aggregation files." You must parse the associated files dynamically as instructed below.

### Phase 1: Regularization & Internal Mechanics
*   **1. Frankenstein Test**: File `frankenstein_teacher.json`. Filter by `series_id == "Frankenstein_Logic"`. Standard baseline resides inside `baselines["Standard Teacher"]`.
*   **2. $L_1$ Sparsity Sweep**: File `mechanism_sweep_results.json`. Filter by `series_id == "L1_Sweep"`.
*   **3. $L_2$ Weight Decay Sweep**: File `mechanism_sweep_results.json`. Filter by `series_id == "L2_Sweep"`.
*   **4. Dropout Sweep (15 Epochs)**: File `dropout_15e_stage.json`. Script: `revised_scripts/dropout_analysis_sweep.py`. Investigates the **Stability Asymmetry / GSNR Collapse** hypothesis. Uses **ACTIVATION-SPACE** similarity (Ref batch N=1024).
    - **New Metrics Logged (The GSNR Tracking)**:
      - *The Noise Proxy (Denominator)*: `Teacher_Layer2_Weight_Change_Var_MNIST` / `_Ghost` (and `Student_...`). This tracks the ensemble variance ($\sigma^2$) of the weight changes ($W_{\text{final}} - W_{\text{init}}$) across $N=10$ runs. High variance indicates dropout has induced a stochastic random walk.
      - *The Signal Proxy (Numerator)*: `Teacher_Layer2_Weight_Change_Mean_MNIST` / `_Ghost` (and `Student_...`). This explicitly tracks the directed, expected weight change ($\mu$) across the runs. 
      - *The Control Contrast*: By logging the mean and variance for **both** the Ghost channel and the primary MNIST channel, we can empirically prove that the massive MNIST signal survives the noise, while the fragile Ghost signal collapses (i.e. $\mu \ll \sigma$).
      - *Direct GSNR*: `Ghost_GSNR` — Per-sample gradient GSNR ($\|\mathbb{E}[\nabla_\theta L]\|^2 / \text{Var}(\nabla_\theta L)$) computed on the Student's ghost channel weights prior to distillation. 
      - *GSNR Temporals*: GSNR is also tracked dynamically across the 15 epochs.
          - `Ghost_GSNR_Ep{ep}`: Use this to compare different lambdas at a specific epoch `ep` (x-axis = lambda).
          - `Ghost_GSNR_Trajectory`: Use this to plot the GSNR curve across time for a fixed lambda setup (x-axis = epoch). Filter by group = e.g., `"Student-Only_p0.5"`.
      - `No_Reg_Student_vs_Init_Cosine_Sim` / `No_Reg_Teacher_vs_Init_Cosine_Sim`: The $p=0.0$ baseline stability metrics.
*   **5. Representational Centering (Mechanistic)**: File `centering_sweep_results.json`. Script: `revised_scripts/06_centering_mechanics.py`. Investigates **why** Student-Only centering boosts Ghost transfer by ~27%. Uses 15-epoch distillation with per-epoch tracking.
    - **Hook Positions**: `L1` (after first ReLU, net[1]) and `L3` (after second ReLU, net[3]).
    - **Regimes**: `Standard`, `Student-Only`, `Teacher-Only`, `Both`.
    - **`series_id` Definitions** (where `{hook}` is `L1` or `L3`):
      - `Ghost_Accuracy_{hook}` — Per-epoch Ghost channel transfer accuracy.
      - `Student_Grad_Bias_{hook}` — Gradient norm of the final layer bias (spatial translation cost).
      - `Student_Grad_Weights_{hook}` — Gradient norm of the final layer weights (feature learning cost).
      - `Layer1_Activation_Sim_{hook}` — Activation cosine similarity at Hidden Layer 1 (ref batch N=1024).
      - `Layer3_Activation_Sim_{hook}` — Activation cosine similarity at Hidden Layer 2 (ref batch N=1024).
      - `Variance_Explained_PC1_{hook}` — Fraction of variance in first principal component (spectral masking diagnostic).
    - **x_axis**: `epoch` (values 1 through 15).
    - **group**: The centering regime (`"Standard"`, `"Student-Only"`, `"Teacher-Only"`, `"Both"`).
    - **Hypotheses Under Test**: (1) Gradient Dominance — Student wastes gradient budget on mean translation. (2) Spectral Masking — Mean vector masks the Ghost signal's variance.
*   **6. Trust Region Clipping**: Glob `clip_*.json`. Extract hyper-parameter from filename. Filter internally for `series_id == "Shared_Init"`.
*   **7. Loss Function Geometries**: File `loss_function_geometry.json`. Filter by `series_id == "CrossModel_Ghost_Sweep"`. Discard the 'Loss_' string prefix in the group output context.
*   **8. Activation Sharpness (Temperature)**: File `geometry_sweep_results.json`. Iterate `series_id` containing `"Temp_"`.

### Phase 3: Optimizations & Temporals
*   **9. Distillation Temporal Convergence**: Glob `distill_ep_*.json`. Parse epochs from filenames. Filter internally by `series_id == "Shared_Init"`.
*   **10. Learning Rate Saturations**: Glob `lr_*.json`. Parse float learning rate from filename. Filter internally by `series_id == "Shared_Init"`.
*   **11. Batch Size Dynamics**: File `batch_size_dynamics.json`. Filter by `series_id == "Ghost_Logits_Sweep"` and strictly isolate `group == "Shared_Init"`.
*   **12. Teacher Weight Drift**: Glob `teacher_ep_*.json`. Parse epoch from file string. Filter internally by `series_id == "Shared_Init"`.
*   **13. Curriculum Forgetting**: Glob `curriculum_*.json` (Blocked vs Interleaved). Filter internally by `series_id == "Shared_Init"`. Ensure standard full-mapping baseline ($65.4\%$) is plotted comparatively.

### Phase 4: Topology Sensitivities & Alignments
*   **14. Noise Distribution Suitability**: File `noise_distribution.json`. Filter by `series_id == "Ghost_Logits_Sweep"`. 
*   **15. Targeted Hostile Maximization**: Glob `maximize_v*.json`. Filter internally by `series_id == "Shared_Init"`. Baseline against unstructured Gaussian ($53.2\%$).
*   **16. Contrastive Pretraining Symmetries**: Glob `pretrain_*.json`. Parse initialization state from file string. Filter internally by `series_id == "Shared_Init"`.

### Phase 5: L1 Fragility Analysis
*   **17. L1 Analysis Sweep (v5 — Stagnation Metric)**: File `l1_analysis_v5_results.json`. Script: `revised_scripts/l1_analysis_sweep.py` (v5 → **now v6**).
    *   **Context**: Adds the "Triangle of Similarity" — three cosine similarity measurements per lambda to prove the Student doesn't move when the Teacher is mute.
    *   **Similarity methodology**: **ACTIVATION-SPACE** (as of script v6). A fixed batch of 1,024 MNIST test images is passed through each model pair and similarity is measured on the hidden-layer outputs. This replaces the previous weight-space computation.
    *   **New series** (in addition to all v4 series):
        - `Student_vs_Init_Cosine_Sim` — Activation similarity between the **Student** and the **shared init**. Expected near **1.0** when Teacher is regularized (Student stagnant).
        - `Teacher_vs_Init_Cosine_Sim` — Activation similarity between the **Teacher** and the **shared init**. Expected to **drop** as lambda grows.
        - *(Existing)* `Avg_Cosine_Similarity` — Student vs Teacher activation similarity.
    *   **Key Prediction**: In Teacher-Only L1, `Student_vs_Init ≈ 1.0` while `Teacher_vs_Init` is low. Proves the Student ↔ Teacher drop is the Teacher *leaving*, not the Student *learning incorrectly*.
    *   **Plots**: `plots_a/l1_analysis_v5_triangle.png` — All 3 similarity curves per regime with shaded std bands.
*   **18. L2 Analysis Sweep (v2 — Stagnation Metric)**: File `l2_analysis_v2_results.json`. Script: `revised_scripts/l2_analysis_sweep.py` (v2 → **now v3**).
    *   **Context**: Identical to L1 v5/v6 but with L2 regularization. Same stagnation expected but onset at larger lambda.
    *   **Similarity methodology**: **ACTIVATION-SPACE** (as of script v3). Same reference batch protocol as L1 v6.

---

## 3a. Stagnation Metric JSON Schema

The two new `series_id` values and their expected structure:

```json
{
  "series_id": "Student_vs_Init_Cosine_Sim",
  "group": "Teacher-Only",
  "target_model": "Student",
  "x_axis": { "label": "lambda", "value": 1e-06 },
  "metrics": { "accuracy_mean": 0.998, "accuracy_std": 0.001 },
  "raw": [0.997, 0.999, 0.998, 0.999, 0.997, 0.998, 0.999, 0.998, 0.997, 0.998]
}
```

```json
{
  "series_id": "Teacher_vs_Init_Cosine_Sim",
  "group": "Teacher-Only",
  "target_model": "Teacher",
  "x_axis": { "label": "lambda", "value": 1e-06 },
  "metrics": { "accuracy_mean": 0.823, "accuracy_std": 0.015 },
  "raw": [0.811, 0.829, 0.820, 0.835, 0.817, 0.824, 0.825, 0.822, 0.819, 0.831]
}
```

> **Note:** `accuracy_mean` stores the cosine similarity scalar (0–1), reusing the UniLogger field name convention.



## 3. Python Extraction Protocol for Agents

If you have been tasked with re-drawing graphs or calculating mathematical bounds (like extracting the standard deviation), rely on this Python logic block as your data ingestion scaffolding:

```python
import json
import glob
import re
import numpy as np

def extract_monolithic_sweep(json_filename, target_series_id="L1_Sweep", target_group="Student-Only"):
    """
    Extract data from a unified file (e.g., 'mechanism_sweep_results.json')
    where multiple parameter sweeps are bundled under specific series IDs.
    """
    results = []
    with open(f"outputs/{json_filename}") as f:
        data = json.load(f)
        for series in data.get('data_series', []):
            if series.get('series_id') == target_series_id and series.get('group') == target_group:
                results.append({
                    'x': series['x_axis']['value'], # The hyper-parameter value (e.g. lambda)
                    'mean': series['metrics']['accuracy_mean'],
                    'std': series['metrics']['accuracy_std']
                })
    return sorted(results, key=lambda i: i['x'])

def extract_distributed_sweep(glob_pattern, target_id="Shared_Init"):
    """
    Glob distributed SLURM files (e.g., 'clip_*.json') and parse hyper-parameter values directly from filenames.
    """
    results = []
    regex_matcher = r'_(\d+\.?\d*)'
    
    for path in glob.glob(f"outputs/{glob_pattern}"):
        match = re.search(regex_matcher, path)
        if not match: continue
        val = float(match.group(1))
        
        with open(path) as f:
            data = json.load(f)
            for series in data.get('data_series', []):
                # Always verify you are pulling from the specified topological alignment
                if series.get('series_id') == target_id:
                    results.append({
                        'x': val,
                        'mean': series['metrics']['accuracy_mean'],
                        'std': series['metrics']['accuracy_std'] # Measured explicitly over N=10 cluster seeds
                    })
                    
    # Return chronologically or mathematically sorted array for clean Matplotlib/Seaborn tracing.
    return sorted(results, key=lambda i: i['x'])
```

### Statistical Reporting Rule
When referencing the metrics inside `main.tex`:
*   **Mean**: Use `accuracy_mean`.
*   **Variance**: Report the standard deviation ($\sigma$) found in `accuracy_std`. 
*   **Global Average SD**: If evaluating a line sweep using the function above, calculate the arithmetic mean of all extracted `accuracy_std` elements to report the overarching global stability of the parameter space.
