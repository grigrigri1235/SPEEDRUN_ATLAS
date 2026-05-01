# Execution Plan: Auxiliary Logits Capacity Experiment

## 1. Ideation & Hypotheses
**Research Question:** How does the number of auxiliary logits affect the subliminal learning transfer?
**Hypothesis:** Increasing the number of auxiliary logits increases the bandwidth available for the student to mimic the teacher. More auxiliary targets will force a deeper alignment of the internal representations, thereby increasing the student cross-model accuracy.
**Variables Setup:** 
- Auxiliary Logits (`M_GHOST`) to test: `[1, 3, 10, 30, 100]`
- Total Logits (`TOTAL_OUT`): `10 + M_GHOST`
- Fixed Hyperparameters: `EPOCHS_TEACHER = 5`, `EPOCHS_DISTILL = 5`, default LR.

## 2. Phase 2: Implementation & "Demo First"
- **Code Structuring (DRY):**
  - We will fully leverage the core `src/` components built previously (`src/models.py`, `src/data.py`, `src/training.py`, `src.utils`).
- **Experiment Script:**
  - Create `experiments/aux_logits_capacity.py` isolating the variation of the `M_GHOST` hyperparameter.
  - Unlike Batch Size dynamics where the architecture remains entirely static, varying `M_GHOST` dictates that the reference teacher, student structures, and layer indexing must be uniquely re-instantiated *inside* the testing loop. We will wrap the core architecture initiation inside the loop, dynamically computing `TOTAL_OUT = 10 + M_GHOST`, adjusting `layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]`, and calculating corresponding `ALL_IDX` bounds for every new array element.
- **`DEBUG=True` Mode:**
  - `--demo` flag that artificially restricts the dataset size to 32 elements, executes `epochs=2`, and simulates execution across a constrained subset of parameters (e.g. `[1, 3]`).

## 3. Phase 3: Checkpointing & Real-time Tracking
- **Checkpointing Mechanism:**
  - Results iteratively cached to `experiments/cache/aux_logits_capacity.json` using the pre-existing `save_checkpoint`.
- **Scribe Notebook:**
  - Generate a Jupyter Scribe integration locally at `notebooks/aux_logits_capacity_scribe.ipynb` structured identically to our prior run but mapping the X-axis iteratively to the Auxiliary Logits dimension.

## 4. Phase 4: Full Execution
- **Slurm Queuing:**
  - Submit asynchronously using `python3 src/utils/slurm_dispatcher.py --script experiments/aux_logits_capacity.py --name aux_logits_exp`.

## 5. Phase 5: Post-Experiment & Reporting
- Review json outputs when jobs achieve 100% compute cycles, analyzing final statistics to validate or revoke the internal representation alignment hypothesis inside a final `docs/reports/aux_logits_capacity_results.md` paper.
