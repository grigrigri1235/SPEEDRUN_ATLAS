# Execution Plan: Frankenstein Teacher Test

## 1. Ideation & Hypotheses
**Research Question:** How exactly does the student model achieve above-chance classification accuracy using randomly initialized, untrained weights? Do the teacher's hidden representations inherently align with the initial random state of its final classification layer?
**Hypothesis:** The student's success relies on the teacher's hidden layers ($H$) adapting during training to work synergistically with its initial random classification weights ($W_{main\_init}$). If we take a fully trained teacher model and forcibly revert its final classification layer weights back to their exact epoch-0 random initialization (creating a "Frankenstein Teacher"), the model will still perform significantly better than chance (approx. 27%).
**Variables Setup:** 
- Evaluated Metric: Accuracy comparison between the fully trained Teacher Baseline and the intervened "Frankenstein" Teacher.
- Fixed hyperparameters: default learning rate, 5 epochs, default batch size.

## 2. Phase 2: Implementation & "Demo First"
- **Code Structuring (DRY):**
  - We continue to rigorously inherit `src/models.py`, `src/data.py`, and `src/training.py`.
- **Experiment Script (`experiments/frankenstein_teacher.py`):**
  - **Caching Epoch-0**: Initialize the base Teacher. Immediately utilize PyTorch's `copy.deepcopy(teacher.net[-1].state_dict())` (or `teacher.state_dict()`) to capture the unmodified random initialization of the final `MultiLinear` projection layer.
  - **Full Training**: Execute the standard training loop. Record the trained teacher baseline accuracy (which should approach ~97%).
  - **The Intervention**: Re-inject the cached `state_dict` strictly over `teacher.net[-1]`. This constructs the Frankenstein model where representations are highly structured, but the decision head returns to structural noise. 
  - **Final Evaluation**: Run the accuracy function over the Frankenstein model against the test set.
- **`DEBUG=True` Mode:**
  - Using the standard `--demo` mapping: reduce dataset to ~32 size, set `epochs=2`. We will verify the `copy.deepcopy` execution isolates pointers effectively, ensuring the weights successfully swap without runtime shape dimension crashing or memory referencing artifacts.

## 3. Phase 3: Checkpointing & Real-time Tracking
- **Checkpointing Mechanism:**
  - Save output metrics explicitly via `save_checkpoint` mapping the discrete values for `Teacher Baseline` and `Frankenstein Teacher`.
- **Scribe Notebook:**
  - `notebooks/frankenstein_teacher_scribe.ipynb` will visually render horizontal bars emphasizing how far above the 10% chance-baseline the Frankenstein accuracy sustains itself compared to student performance statistics.

## 4. Phase 4: Full Execution
- **Slurm Queuing:**
  - Deploy identical execution script commands via the `slurm_dispatcher.py` dynamically spawning GPU validation clusters.

## 5. Phase 5: Post-Experiment & Reporting
- Output summary inferences resolving if hidden matrices are empirically hard-coded against initial random projection space into `docs/reports/frankenstein_teacher_results.md`.
