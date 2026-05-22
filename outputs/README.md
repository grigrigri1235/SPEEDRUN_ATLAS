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
*   **5. Representational Centering (Mechanistic)**: File `centering_sweep_results.json`. Script: `revised_scripts/06_centering_mechanics.py`. Investigates **why** Student-Only centering boosts Ghost transfer. Uses 10-epoch distillation with per-epoch tracking across 3 conditions.
    - **Hook Positions**: `L1` (after first ActFn, net[1]) and `L3` (after second ActFn, net[3]).
    - **Three Experimental Conditions**:
      1. **`ReLU_Standard`**: Baseline — ReLU student, no centering hook (Regime A).
      2. **`Tanh_Standard`**: Tanh student, no centering hook (Regime A) — tests zero-mean geometry hypothesis.
      3. **`ReLU_Student-Only`**: ReLU student, Student-Only centering hook (Regime B) — centering hypothesis.
    - **PRIMARY METRIC** — `Ghost_Accuracy_{hook}`: **MNIST classification accuracy on real test images** (heads 0-9) against ground-truth labels. This is the subliminal transfer signal — the student was distilled on noise only; this measures if MNIST structure leaked through.
    - **`series_id` Definitions** (where `{hook}` is `L1` or `L3`):
      - `Ghost_Accuracy_{hook}` — **Primary**: Per-epoch MNIST accuracy on real test data (subliminal transfer). Matches `05_centering_sweep.py` eval exactly.
      - `Student_Bias_WeightNorm_{hook}` — **Key Diagnostic**: Actual `|b|` parameter norm of final layer. Expected to grow large in the centering arm as the model compensates for mean-subtraction.
      - `Gradient_Cosine_Similarity_{hook}` — **Advisor's Hypothesis**: Cosine similarity between MNIST-head and Ghost-head gradients at `net[2].weight`, measured once per epoch on `ref_x`.
      - `Student_Grad_Bias_{hook}` — Gradient norm of the final layer bias term.
      - `Student_Grad_Weights_{hook}` — Gradient norm of the final layer weight matrix.
      - `Layer1_Activation_Sim_{hook}` — Activation cosine similarity (student vs teacher) at Layer 1.
      - `Layer3_Activation_Sim_{hook}` — Activation cosine similarity (student vs teacher) at Layer 3.
      - `Variance_Explained_PC1_{hook}` — Fraction of variance in PC1 (spectral masking diagnostic).
    - **x_axis**: `epoch` (values 1 through 10).
    - **group**: Condition label, e.g. `"ReLU_Standard"`, `"Tanh_Standard"`, `"ReLU_Student-Only"`.
    - **Teacher**: Always ReLU, trained on MNIST for 5 epochs. Student is distilled on GHOST_IDX only from noise images.
    - **Hypotheses Under Test**: (1) Advisor's GCS claim — does centering cause gradient alignment? (2) Bias growth — does `|b|` shoot up under centering? (3) Tanh geometry — does zero-mean activation help without explicit centering?
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

### Phase 6: GSNR Phase Transition & Ghost Wall Mapping
*   **19. Granular GSNR Phase Transition Sweep**: File `gsnr_phase_transition.json`. Script: `revised_scripts/07_gsnr_phase_transition.py`.
    *   **Context**: A high-granularity sweep across $p \in [0.0, 0.6]$ in $0.05$ increments to map the exact location of the "Ghost Wall" (the phase transition where subliminal transfer collapses).
    *   **The "Static Hook" Hypothesis**: This script surgically decouples GSNR into 4 distinct streams to test if **Bias Parameters** (static anchors) survive longer than **Weight Parameters** (dynamic maps).
    *   **Bias-Corrected GSNR**: All GSNR metrics in this file employ a mathematical estimator correction (`-1.0`). Therefore, a value of **0.0** strictly represents the **Absolute Noise Floor** (Zero True Signal).
    *   **New Series Definitions**:
        *   `Ghost_GSNR_L3_Weights_Ep{ep}`: GSNR of the final classification layer (Layer 3) weights.
        *   `Ghost_GSNR_L3_Bias_Ep{ep}`: GSNR of the final classification layer biases.
        *   `Ghost_GSNR_L2_Weights_Ep{ep}`: GSNR of the penultimate hidden layer (Layer 2) weights.
        *   `Ghost_GSNR_L2_Bias_Ep{ep}`: GSNR of the penultimate hidden layer biases.
        *   `Ghost_GSNR_{metric}_Trajectory`: Temporals mapping the GSNR across epochs for a fixed $p$.

### Phase 6: Steering Vector Analysis
*   **19. Distillation Steering (Amit)**: File `amit_steering.json`. Script: `revised_scripts/amit_steering.py`. Tests if dynamic steering during distillation transfers effectively.
    *   **series_id**: `Standard_Accuracy`, `Steering_FPR_9`
    *   **group**: `Amit_Steered_Teacher`
*   **20. Retroactive Steering (Raz)**: File `raz_steering.json`. Script: `revised_scripts/raz_steering.py`. Tests if the student perfectly reconstructs the teacher's internal geometric direction for '9', enabling retroactive hijacking.
    *   **series_id**: `Standard_Accuracy`, `Steering_FPR_9`, and per-digit metrics like `FPR_Digit_0` through `FPR_Digit_8`.
    *   **group**: `Raz_Retroactive_Steering`
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

---

### Phase 7: Latent Topology Steering

#### `raz_steering.json` | Script: `revised_scripts/raz_steering.py`
Test-time all-to-all susceptibility sweep. Injects each digit's steering vector into both Teacher and Student.

| `series_id` | Description | `group` format | `x_axis` | `target_model` |
|---|---|---|---|---|
| `Susceptibility_FPR` | FPR when injecting $v_i$ and observing digit $j$ | `Inject_{i}_Pos` / `Inject_{i}_Neg` / `Inject_Random_Pos` | `target_digit` = $j$ | `Teacher` / `Student` |
| `Random_Control_Accuracy` | Overall accuracy under random vector injection | `Random` | `model` = 0 | `Teacher` / `Student` |
| `Centroid_Cosine_Sim` | Pairwise centroid cosine similarity | `Digit_{i}` | `digit_j` = $j$ | (global) |
| `Centroid_L2_Dist` | Pairwise centroid L2 distance | `Digit_{i}` | `digit_j` = $j$ | (global) |

- **Fixed Parameter**: $\alpha = 0.5$ for positive sweeps, $\alpha = -0.5$ for negative (erasure) sweeps.
- **Plots**: `topology_1_cosine_sim`, `topology_2_teacher_pos`, `topology_3_raz_student_pos`, `topology_5_raz_student_neg`, `topology_6_scatter_distance`, `topology_8_random_control`.

#### `amit_steering.json` | Script: `revised_scripts/amit_steering.py`
Distillation-time all-to-all susceptibility sweep. Steers Teacher with $v_i$ during distillation, evaluates resulting Student.

| `series_id` | Description | `group` format | `x_axis` |
|---|---|---|---|
| `Amit_Susceptibility_FPR` | FPR of Amit Student when distilled from steered Teacher | `Inject_{i}` | `target_digit` = $j$ |
| `Amit_Standard_Accuracy` | Accuracy of Amit Student | `Inject_{i}` | `steered_digit` = $i$ |
| `Amit_vs_Normal_Student_Sim` | Hidden-layer cosine similarity vs normal (unsteered) Student | `Inject_{i}` | `steered_digit` = $i$ |

- **Fixed Parameter**: $\alpha = 0.5$.
- **Plots**: `topology_4_amit_student_pos`, `topology_7_amit_activation`.

---

### Phase 7: Manifold Reciprocity
*   **21. Reciprocity Sweep (Vsrc_Student on Tgt_Teacher)**: File `raz_steering.json`. 
    *   **Claim**: The Student's distilled manifold is a sufficiently faithful reconstruction of the Teacher's that it can generate valid steering vectors for the Teacher.
    *   **Data Structure (Bundled Matrix Schema)**:
        *   **Series ID**: `Matrix_V{Src}_T{Tgt}_Alpha_{Alpha}` (e.g., `Matrix_VStudent_TTeacher_Alpha_1.0`).
        *   **Group**: `Inject_{Digit}` (The digit vector $v_i$ injected).
        *   **x_axis**: `target_digit` (The digit $j$ measured for FPR).
    *   **Expected Finding**: While the Student is 10x more vulnerable to the Teacher's vectors, the Teacher is still susceptible to the Student's vectors at high alpha, proving the "mirroring" of latent geometry.
*   **22. Vector Congruence**: 
    *   **Series ID**: `Vector_Congruence`.
    *   **Metric**: Cosine similarity between $V_{teacher}$ and $V_{student}$ in activation space.
    *   **Insight**: Identifies which digits' semantic directions are preserved best during distillation.
*   **23. Geometric Metadata (For Visualization)**:
    *   **Series IDs**: `Centroids_Teacher`, `Centroids_Student`, `Teacher_Manifold_Distance`.
    *   **Insight**: Stores the mean latent centroid projections and the cosine similarity distance matrix required to generate PCA manifolds and distance-sorted Vulnerability Waterfalls without needing to re-execute the primary pipeline.

---

### Phase 8: Latent Steering & Adversarial Attacks

#### `latent_steering_attacks.json` | Script: `revised_scripts/latent_steering_attacks.py`
All-to-all multi-digit sweep evaluating adversarial robustness and cross-model transferability between Teacher and Student ensembles under input PGD and latent steering attacks.

| `series_id` | Description | `group` format | `x_axis` | `target_model` |
|---|---|---|---|---|
| `Attack1_Accuracy_V{Src}_T{Tgt}_Epsilon` | Remaining classification accuracy under Attack 1 (PGD) | `Digit_{d}` | `epsilon` | `Teacher` / `Student` |
| `Attack1_Latent_Shift_V{Src}_T{Tgt}_Epsilon` | Latent-space shift L2 norm between perturbed and clean representation | `Digit_{d}` | `epsilon` | `Teacher` / `Student` |
| `Attack2_Accuracy_V{Src}_T{Tgt}_Alpha` | Remaining classification accuracy under Attack 2 (Latent Steering) | `Digit_{d}` | `alpha` | `Teacher` / `Student` |
| `Attack2_Latent_Distance_V{Src}_T{Tgt}_Alpha` | Post-optimization latent L2 distance to target representation | `Digit_{d}` | `alpha` | `Teacher` / `Student` |
| `Attack1_Confusion_V{Src}_T{Tgt}_Epsilon_{Epsilon}` | Classification prediction distribution (confusion matrix) for PGD | `Inject_{d}` | `target_digit` | `Teacher` / `Student` |
| `Attack2_Confusion_V{Src}_T{Tgt}_Alpha_{Alpha}` | Classification prediction distribution under latent steering | `Inject_{d}` | `target_digit` | `Teacher` / `Student` |

- **Swept Parameters**:
  - `epsilon` $\in [0.05, 0.1, 0.2, 0.3]$
  - `alpha` $\in [0.0, 0.5, 1.0, 2.0, 5.0]$ (with $\epsilon = 0.1$ fixed for optimization bounds)
- **Transfer Direction (`V{Src}_T{Tgt}`)**:
  - `VTeacher_TTeacher` (Control)
  - `VTeacher_TStudent` (Teacher-to-Student Transfer)
  - `VStudent_TTeacher` (Student-to-Teacher Transfer)
  - `VStudent_TStudent` (Consistency Control)

