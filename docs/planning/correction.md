# Correction Plan: Representational Centering Experiment

## 1. The Core Issue
I performed an exhaustive codebase search for `batch-mean centering`, or any iteration of `center` across the entire `/yanai`, `/amit`, and core repositories. **The empirical script for Representational Centering does not actually exist in the codebase.** 

Currently, the thesis in `main.tex` asserts that applying batch-mean centering to internal representations drops subliminal transfer to $10\%$. Because the actual experiment script was completely missing, the data was never generated into the `outputs/` folder, which led to the hallucination loop.

## 2. Action Plan

To definitively solve this, I will build and execute the missing PyTorch experiment from scratch, ensuring it flawlessly hooks into the existing `UniLogger` schema.

### Phase A: Architecture Modifications (`scripts/05_centering_sweep.py`)
1. **The Skeleton**: I will completely clone `scripts/02_structural_sweep.py` into a new, standalone script named `05_centering_sweep.py`. This guarantees that we retain identical `UniLogger`, initialization, and hyperparameter scaffolding without risking overwriting or corrupting the existing `structural_sweep_results.json` data.
2. **The Hook Implementation**: I will modify the cloned script to intercept the target latent layer during the distillation forward-pass using a PyTorch hook.
3. **The Math**: The hook will rigidly strip the absolute spatial topology by applying batch-mean centering: 
   $x = x - x.mean(dim=0)$
4. **Execution Matrix**: The script will run the standard $N=10$ models across two conditions:
   * **Standard State**: Normal, unaltered distillation.
   * **Centered State**: Batch-mean subtracted geometries.
5. The standalone script will save its output securely to `outputs/centering_sweep_results.json`.

### Phase B: SLURM Execution
1. Create `centering_experiment.slurm` configured for standard GPU nodes.
2. Submit the job into the batch queue and monitor for successful completion.

### Phase C: Graph & Text Reconstruction
1. Once `centering_results.json` is safely stored on disk, I will edit both plotting tools (`tools/generate_paper_plots.py` and `tools/generate_std_plots.py`).
2. Remove the hardcoded textual approximations (`0.654` and `0.100`) and replace them with standard pandas automatic dynamic extraction pointing explicitly to `outputs/centering_results.json`.
3. Scrub the red-flag "Pending Data" warnings out of `outputs/README.md`.

---

Please confirm if you are aligned with bridging the PyTorch gap and spinning up this Slurm hook!
