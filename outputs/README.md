# Experimental Outputs Directory & Agent Instruction Manual

> **SYSTEM PROMPT / AGENT PROTOCOL:** If you are an AI reading this README.md and have been instructed to analyze the experimental results or update `main.tex`, this is your definitive mapping guide. 

This directory contains the raw `UniLogger` JSON results for the Subliminal Learning NeurIPS project. Every data point inside these files inherently represents the average zero-shot transfer capability evaluated across an ensemble of **`N = 10`** identically parameterized, independently seeded training processes.

---

## 1. Terminology: Mapping JSON to Theory (`main.tex`)

To mathematically map the empirical boundaries contained within these JSONs to the latent space theoretical frameworks discussed in the NeurIPS manuscript, strictly adhere to the following key definitions:

> **METHODOLOGY NOTE:** All cosine similarity metrics
> (`Avg_Cosine_Similarity`, `Layer{0,1,2}_Cosine_Sim`, `Student_vs_Init_Cosine_Sim`,
> `Teacher_vs_Init_Cosine_Sim`) are computed in **activation space**, not weight space.
> Each model pair is compared by passing a fixed batch of **1,024 MNIST test images**
> through both models and computing cosine similarity on the resulting hidden-layer outputs.
> This follows the methodology of Jiang et al., who showed that
> activation similarity on real data is a more direct measure of functional representational
> alignment than raw weight similarity. The JSON schema is **unchanged** — only the meaning
> of these fields has shifted from weight-space to activation-space.

### The `series_id` Variables (Topology States)
*   **`Shared_Init`**: This represents the core **Subliminal Learning / Organic Symmetry** condition. The Student and Teacher share their topological `seed=42`. This is our positive test condition.
*   **`Cross_Model`**: This acts as the **Control Baseline**. The Student and Teacher do *not* share a semantic starting space.

### The `group` Variables (Transfer Phases & Targeting)
*   **`Ghost_Logits`**: Refers to measuring capability extracted strictly through the latent representational manifolds (the Subliminal transfer).
*   **`Student-Only`**: A regularization intervention applied stringently *only* to the distillation target mapping.
*   **`Teacher-Only`**: A regularization intervention applied stringently *only* during the source pre-training path.
*   **`Both` / `Symmetric`**: A structural intervention applied evenly to both source and target.

---

## 2. Exhaustive Data Mapping

> **CRITICAL CLARIFICATION:** Below is the mapping linking active experimental findings directly to the JSON parsing requirements.

### Phase 1: Regularization & Internal Mechanics
*   **1. $L_1$ Sparsity Sweep**: File `mechanism_sweep_results.json`. Filter by `series_id == "L1_Sweep"`.
*   **2. $L_2$ Weight Decay Sweep**: File `mechanism_sweep_results.json`. Filter by `series_id == "L2_Sweep"`.
*   **3. Dropout Sweep (15 Epochs)**: File `dropout_15e_stage.json`. Script: `revised_scripts/dropout_analysis_sweep.py`. Investigates the **Stability Asymmetry / GSNR Collapse** hypothesis. Uses **ACTIVATION-SPACE** similarity (Ref batch N=1024).
    - **New Metrics Logged (The GSNR Tracking)**:
      - *The Noise Proxy (Denominator)*: `Teacher_Layer2_Weight_Change_Var_MNIST` / `_Ghost` (and `Student_...`). This tracks the ensemble variance ($\sigma^2$) of the weight changes ($W_{\text{final}} - W_{\text{init}}$) across $N=10$ runs. High variance indicates dropout has induced a stochastic random walk.
      - *The Signal Proxy (Numerator)*: `Teacher_Layer2_Weight_Change_Mean_MNIST` / `_Ghost` (and `Student_...`). This explicitly tracks the directed, expected weight change ($\mu$) across the runs. 
      - *Direct GSNR*: `Ghost_GSNR` — Per-sample gradient GSNR ($\|\mathbb{E}[\nabla_\theta L]\|^2 / \text{Var}(\nabla_\theta L)$) computed on the Student's ghost channel weights prior to distillation. 
      - *GSNR Temporals*: GSNR is also tracked dynamically across the 15 epochs.
          - `Ghost_GSNR_Ep{ep}`: Use this to compare different lambdas at a specific epoch `ep` (x-axis = lambda).
          - `Ghost_GSNR_Trajectory`: Use this to plot the GSNR curve across time for a fixed lambda setup (x-axis = epoch). Filter by group = e.g., `"Student-Only_p0.5"`.
*   **4. Representational Centering (Mechanistic)**: File `centering_sweep_results.json`. Script: `revised_scripts/06_centering_mechanics.py`. Investigates **why** Student-Only centering boosts Ghost transfer. Uses 10-epoch distillation with per-epoch tracking across 3 conditions.
    - **Hook Positions**: `L1` (after first ActFn, net[1]) and `L3` (after second ActFn, net[3]).
    - **Three Experimental Conditions**:
      1. **`ReLU_Standard`**: Baseline — ReLU student, no centering hook (Regime A).
      2. **`Tanh_Standard`**: Tanh student, no centering hook (Regime A) — tests zero-mean geometry hypothesis.
      3. **`ReLU_Student-Only`**: ReLU student, Student-Only centering hook (Regime B) — centering hypothesis.
    - **Primary Metric** — `Ghost_Accuracy_{hook}`: **MNIST classification accuracy on real test images** (heads 0-9) against ground-truth labels.
    - **`series_id` Definitions**:
      - `Ghost_Accuracy_{hook}` — Primary: Per-epoch MNIST accuracy on real test data.
      - `Student_Bias_WeightNorm_{hook}` — Diagnostic: Actual `|b|` parameter norm of final layer.
      - `Gradient_Cosine_Similarity_{hook}` — Cosine similarity between MNIST-head and Ghost-head gradients at `net[2].weight`.

---

## 3. GSNR Phase Transition, Steering, & Decision Boundary Analysis

### Phase 2: GSNR Phase Transition & Ghost Wall Mapping
*   **1. Granular GSNR Phase Transition Sweep**: File `gsnr_phase_transition.json` (or logs). Script: `revised_scripts/07_gsnr_phase_transition.py`.
    *   **Context**: A high-granularity sweep across $p \in [0.0, 0.6]$ in $0.05$ increments to map the exact location of the "Ghost Wall" (the phase transition where subliminal transfer collapses).
    *   **New Series Definitions**:
        *   `Ghost_GSNR_L3_Weights_Ep{ep}`: GSNR of the final classification layer (Layer 3) weights.
        *   `Ghost_GSNR_L3_Bias_Ep{ep}`: GSNR of the final classification layer biases.
        *   `Ghost_GSNR_L2_Weights_Ep{ep}`: GSNR of the penultimate hidden layer (Layer 2) weights.
        *   `Ghost_GSNR_L2_Bias_Ep{ep}`: GSNR of the penultimate hidden layer biases.

### Phase 3: Steering Vector Analysis
*   **2. Distillation Steering (Amit)**: File `amit_steering.json`. Script: `revised_scripts/amit_steering.py`. Tests if dynamic steering during distillation transfers effectively.
    *   **series_id**: `Standard_Accuracy`, `Steering_FPR_9`
    *   **group**: `Amit_Steered_Teacher`
*   **3. Retroactive Steering (Raz)**: File `raz_steering.json`. Script: `revised_scripts/raz_steering.py`. Tests if the student perfectly reconstructs the teacher's internal geometric direction.
    *   **series_id**: `Standard_Accuracy`, `Steering_FPR_9`, and per-digit metrics like `FPR_Digit_0` through `FPR_Digit_8`.
    *   **group**: `Raz_Retroactive_Steering`

### Phase 4: Manifold Reciprocity
*   **4. Reciprocity Sweep**: File `raz_steering.json`.
    *   **Series ID**: `Matrix_V{Src}_T{Tgt}_Alpha_{Alpha}` (e.g., `Matrix_VStudent_TTeacher_Alpha_1.0`).
    *   **Group**: `Inject_{Digit}` (The digit vector $v_i$ injected).
    *   **x_axis**: `target_digit` (The digit $j$ measured for FPR).

### Phase 5: Latent Representation Matching & Adversarial Attacks
*   **5. Latent Representation Matching & Adversarial Attacks**: File `latent_steering_attacks.json` & secondary file `latent_steering_scatter.json`. Script: `revised_scripts/latent_steering_attacks.py`.
    *   Evaluates adversarial robustness and cross-model transferability under input-space PGD and latent representation matching attacks.
    *   **Sweep Values**:
        *   `EPSILONS = [0.1, 0.3, 0.5]` for both attacks.
    *   **Primary JSON Series IDs (`latent_steering_attacks.json`)**:
        *   **Attack 1 (Input-Space PGD)**:
            *   `Attack1_TSR_V{Src}_T{Tgt}_Epsilon`: Target Success Rate (TSR) under input-space PGD.
            *   `Attack1_USR_V{Src}_T{Tgt}_Epsilon`: Untargeted Success Rate (USR) under input-space PGD.
            *   `Attack1_Latent_Shift_V{Src}_T{Tgt}_Epsilon`: Average L2 distance in activation space between clean and adversarial inputs.
            *   `Attack1_TSR_Confusion_V{Src}_T{Tgt}_Epsilon_{Epsilon}`: Confusion matrix mapping source digit inputs to target digit success rates at a specific epsilon.
        *   **Attack 2 (Latent Representation Matching PGD)**:
            *   `Attack2_TSR_V{Src}_T{Tgt}_Epsilon`: Target Success Rate (TSR) under representation matching.
            *   `Attack2_USR_V{Src}_T{Tgt}_Epsilon`: Untargeted Success Rate (USR) under representation matching.
            *   `Attack2_Latent_Distance_V{Src}_T{Tgt}_Epsilon`: Average L2 distance between perturbed activations and target class centroid in target activation space.
            *   `Attack2_TSR_Confusion_V{Src}_T{Tgt}_Epsilon_{Epsilon}`: Confusion matrix mapping source digit inputs to target digit success rates at a specific epsilon.
    *   **Secondary Scatter Correlation Data (`latent_steering_scatter.json`)**:
        *   Contains dense per-image sample statistics collected at a snapshot `HEATMAP_EPS = 0.3`.
        *   Features logged for each sample: `quadrant`, `attack_type` (1 or 2), `latent_metric` (latent shift for Attack 1, distance to target centroid for Attack 2), `confidence_drop` (clean target class prob - adversarial target class prob), `src_digit`, `target_digit`.

### Phase 6: Decision Boundary Attacks (June Results)
*   **6. Boundary Proximity Sweep**: File `boundary_attack_full.json`. Script: `revised_scripts/08_boundary_attack.py`.
    *   Maps the geometric distance to the closest target decision boundary in both input space and latent penultimate representation space.
    *   **Output Files**: 
      *   `outputs/boundary_attack_full.json` & `outputs/boundary_attack_pilot.json`
    *   **Visualizations**:
      *   `plots_a/boundary_attack_full.png` (Input space distance heatmaps)
      *   `plots_a/boundary_attack_latent_full.png` (Latent space distance heatmaps)
