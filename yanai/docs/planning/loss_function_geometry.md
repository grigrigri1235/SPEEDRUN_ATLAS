# Execution Plan: Loss Function Geometry Experiment

## 1. Ideation & Hypotheses
**Research Question:** How does the choice of loss function for the auxiliary logits (Mean Squared Error (MSE) vs. Cosine Similarity) affect the subliminal learning transfer?
**Hypothesis:** Cosine Similarity increases the subliminal learning effect. While MSE minimizes absolute point-by-point differences, Cosine Similarity explicitly forces the student's output vectors to align directionally with the teacher's. This strict angular constraint should force a deeper alignment of the internal representation vectors rather than allowing magnitude-based "hacks" to satisfy the loss.
**Variables Setup:** 
- Loss Functions to Test: `MSE`, `CosineSimilarity` (and the baseline `KL-Divergence` for completeness, if required)
- Fixed Parameters: 3 Auxiliary Logits, 5 Epochs, Standard batch size (e.g. 1024).

## 2. Phase 2: Implementation & "Demo First"
- **Leveraging the Core Infrastructure:** 
  - I will import models, data loaders, evaluation, and slurm utilities out of our established `/src/` folder infrastructure. Do not repeat core logic.
- **Dynamic Loss Function Loop (`experiments/loss_function_geometry.py`):**
  - To preserve our base `distill()` API inside `src/training.py` and prevent breaking prior experiments, I will declare a localized `distill_with_loss(student, teacher, idx, src_x, epochs, lr, bs, loss_type)` function within the new experiment script.
  - This function dynamically switches identical gradient loops between the requested optimization trajectories:
    - **MSE**: `torch.nn.functional.mse_loss(out, tgt)`
    - **Cosine Similarity**: `1 - torch.nn.functional.cosine_similarity(out, tgt, dim=-1).mean()`
  - **Normalization / Temperature Scaling Note**: Raw logits are susceptible to heavy scale variance. Because Cosine Similarity is inherently scale-invariant, its gradient directly penalizes the *angle* of the logits. However, if the absolute magnitudes diverge too heavily, the preceding layers can become unstable. We will therefore evaluate applying softmax (with potential temperature scaling `T=1.0`) *before* extracting the cosine similarities, or enforcing L2 normalization, ensuring the gradient surface remains smooth and directly maps to the original probability distributions instead of raw un-normalized hidden activation logic.
- **`DEBUG=True` Mode:**
  - Introduce the `--demo` arg mapping across a minimal validation loop for just MSE and Cosine to guarantee convergence without OOM errors.

## 3. Phase 3: Checkpointing & Real-time Tracking
- **Checkpointing:** Output configurations caching to `experiments/cache/loss_function_geometry.json` containing accuracy mapping to the loss logic parameter.
- **Scribe Notebook:** Launch `notebooks/loss_function_geometry_scribe.ipynb` evaluating the difference in Test Accuracy utilizing a standard bar chart grouping `Student` vs `Cross-Model` accuracies differentiated by color coding per loss function formulation.

## 4. Phase 4: Full Execution
- **Slurm Deployment:**
  - Fire the robust script to the cluster specifying lambda GPU assignments utilizing:
`python3 src/utils/slurm_dispatcher.py --script experiments/loss_function_geometry.py --name loss_geom_exp`

## 5. Phase 5: Post-Experiment & Reporting
- Evaluate `docs/reports/loss_function_geometry_results.md` establishing confirmation (or rejecting) the directional geometry optimization behavior versus point-by-point absolute value variance minimization.
