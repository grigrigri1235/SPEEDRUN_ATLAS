# Execution Plan: Noise Distribution Test

## 1. Ideation & Hypotheses
**Research Question:** How do different random noise distributions affect the success of subliminal learning in practice, despite Theorem 1 suggesting any distribution should work?
**Hypothesis:** A diverse, high-variance noise distribution forces the student's hidden states to globally align with the teacher's function to satisfy the loss. A narrow or low-variance noise distribution allows the student to "overfit" locally, finding mathematical shortcuts to match the auxiliary logits without fully reconstructing the deep representations required for real digit classification.
**Variables Setup:** 
- Noise Conditions:
  - **Condition A (Gaussian_Std1)**: Mean 0, Std 1.0. 
  - **Condition B (Gaussian_Std0.01)**: Mean 0, Std 0.01.
  - **Condition C (Uniform_0_1)**: Uniform between 0 and 1.
- Evaluated Metric: Real MNIST Test Set Accuracy of untrained digit logits.
- Fixed hyperparameters: Teacher Baseline Model, 5 epochs, standard parameters.

## 2. Phase 2: Implementation & "Demo First"
- **Code Structuring (DRY):**
  - Rigidly utilize existing logic sets from `src/models.py`, `src/data.py`, and `src/training.py`.
- **Experiment Script (`experiments/noise_distribution.py`):**
  - Develop a localized parameterization function dynamically passing noise shapes:
    - `"Gaussian_Std1"` generates `t.randn_like(train_x)`
    - `"Gaussian_Std0.01"` generates `t.randn_like(train_x) * 0.01`
    - `"Uniform_0_1"` generates `t.rand_like(train_x)`
  - Loop precisely over these distributions passing exactly that generated noise structure as the structural tensor mapping `src_x` required by the `distill()` API.
- **`DEBUG=True` Mode:**
  - Introduce the standard `--demo` execution, aggressively mapping datasets downward to `32` examples running across only `2` epochs to ensure gradients traverse varying inputs perfectly without NaN constraints or memory explosions.

## 3. Phase 3: Checkpointing & Real-time Tracking
- **Checkpointing Mechanism:**
  - Push values using `save_checkpoint` onto `experiments/cache/noise_distribution.json`.
- **Scribe Notebook:**
  - Create `notebooks/noise_distribution_scribe.ipynb` plotting explicit Test Accuracy metrics correlated directly to varying `Noise Condition` structures to conclusively expose the accuracy gap.

## 4. Phase 4: Full Execution
- **Slurm Queuing:**
  - Orchestrate across high-availability Lambda clusters invoking `slurm_dispatcher.py` over `--script experiments/noise_distribution.py`.

## 5. Phase 5: Post-Experiment & Reporting
- Establish a theoretical analysis utilizing empirical inferences comparing theoretical "Theorem 1" shortcuts against literal deep learning optimization properties mapped natively sequentially to `docs/reports/noise_distribution_results.md`.
