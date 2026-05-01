# Execution Plan: Maximize Subliminal Transfer

## 1. Ideation & Hypotheses
**Research Question:** What drives the magnitude of subliminal learning in practice, and how can we maximize the student's cross-model accuracy given a fixed, narrow auxiliary bandwidth (exactly 3 auxiliary logits)?
**Hypothesis:** Optimization fidelity across a comprehensive input space is the dominant driver. Extending distillation time substantially, forcing stabilization through a Learning Rate Scheduler, scaling optimization metrics firmly through MSE geometry, and blasting the neural logic with High-Variance noise pushes structural mimicry beyond shallow local overfitting to reconstruct incredibly complex entangled global logic.
**Variables Setup:** 
- Auxiliary Logits: Exactly `3`.
- Loss Function: `Mean Squared Error` (Raw Logits Mapping).
- Noise Distribution: `High-Variance Gaussian` (Mean 0, Standard Deviation 3.0).
- Distillation Epochs: `50` (Extended 10x past the 5-epoch baseline). 
- Learning Rate Optimization: `CosineAnnealingLR` dynamically settling boundaries.

## 2. Phase 2: Implementation & "Demo First"
- **Code Structuring (DRY):**
  - Standard MNIST definitions, Accuracy checks, Base Classifiers, and Teacher configurations will be pulled from `/src/`. 
- **Experiment Script (`experiments/maximize_subliminal.py`):**
  - **Dynamic Noise Generator:** Construct explicit boundary logic wrapping `t.randn_like(x) * 3.0`.
  - **`distill_maximizer()`:** We will cleanly execute the logic without mutating existing `src/training.py` codebases by deploying a specifically modified execution loop handling both `F.mse_loss` geometry alongside dynamic updates tracking the `CosineAnnealingLR(optimizer, T_max=epochs)`.
- **`DEBUG=True` Mode:**
  - Introduce identical conditional boundaries routing execution through the `--demo` parameter mapping epochs downward to precisely `2-3` while constraining sizes. The goal evaluates mathematically that the Cosine Annealing scheduler operates identically stepping `.step()` linearly scaling without destroying tensor memories during backward updates.

## 3. Phase 3: Checkpointing & Real-time Tracking
- **Checkpointing Mechanism:**
  - Dump evaluated "Maximizer Student" testing outputs against static standard "Baseline" arrays mapping natively to `experiments/cache/maximize_subliminal.json`.
- **Scribe Notebook:**
  - Create `notebooks/maximize_subliminal_scribe.ipynb` mapping visual evaluation distributions demonstrating exact percentage scaling increases against the previous limits dynamically.

## 4. Phase 4: Full Execution
- **Slurm Queuing:**
  - Hand off the high-impact computation securely to `slurm_dispatcher.py` allowing internal CUDA clusters to sequentially iterate the explicit 50-epoch gradient path identically resolving across the GPU configurations.

## 5. Phase 5: Post-Experiment & Reporting
- Formulate the ultimate conclusion resolving optimal subliminal generation schemas effectively writing final evaluations strictly mapped against mathematical output validations into `docs/reports/maximize_subliminal_results.md`.
