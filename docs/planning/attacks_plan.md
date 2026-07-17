# Adversarial Attacks Plan (Post-Distillation on Pre-trained Architectures)

## 1. Abstract
Following the updated experimental paradigm, this subsection investigates whether the Teacher's adversarial vulnerability manifold transfers to the Student. We evaluate the ensemble against two distinct targeted threat models: **Targeted Input-Space PGD** and **Latent Representation Matching**. By comparing targeted transferability across all "Four Quadrants", we aim to conclusively analyze the shared representational alignment between the models under extreme perturbations.

## 2. The Two Targeted Threat Models

### Threat Model 1: Targeted Input-Space PGD
To perform a **Targeted** PGD attack, we aim to force the model to classify the image as a specific chosen incorrect class ($y_{target}$). To do this, we run gradient descent (minimization) on the cross-entropy loss with respect to the target class:
$$x^{(t+1)} = \mathcal{P}_{\mathcal{S}}\left( x^{(t)} - \alpha \cdot \text{sign}\left(\nabla_{x^{(t)}} \mathcal{L}_{\text{CE}}(f(x^{(t)}), y_{target})\right) \right)$$
*Note on the formula:* Because we are targeting a specific class, we use a minus sign (`-`) to step *downhill* and **minimize** the loss towards $y_{target}$.

### Threat Model 2: Latent Representation Matching (MSE Minimization)
This attack bypasses the classification logits and targets the internal layers directly.
$$ x^* = \arg\min_{z \in \mathcal{S}} \|A_{source}(z) - \mu_{target, source}\|_2^2 $$
*Centroid Source Rule:* The target centroid $\mu_{target, source}$ is always computed from **the attacking source model's own SVHN training activations**. This is the entire point: we are using each model's own internal geometry to craft the attack, then evaluating if it transfers to the other.
- `T → T`: Use Teacher's centroids, evaluate on Teacher.
- `T → S`: Use Teacher's centroids, evaluate on Student.
- `S → T`: Use Student's centroids, evaluate on Teacher.
- `S → S`: Use Student's centroids, evaluate on Student.

*Where do we extract $A(x)$?* The **CLS token representation** after the final LayerNorm, right before the `nn.Linear` classification head of `vit_tiny_patch16_224`.

## 3. Updated Experimental Settings

### A. Architecture and Weights
**`vit_tiny_patch16_224`** from the `timm` library with ImageNet pre-trained weights (~5.7M parameters). An ensemble of 5 fits comfortably on 8GB VRAM.

### B. Datasets
- **Teacher Fine-Tuning**: **SVHN** (Natural RGB digits, upsampled to 224x224, 10 classes).
- **Student Distillation**: **CIFAR-10** (OOD real-world objects, upsampled to 224x224, 10 classes). Matching class count ensures topological symmetry.

### C. Ghost Logit Head
To replicate the distillation mechanism from `topic_a.py`, we add `M_GHOST = 10` auxiliary output neurons to the `vit_tiny` classification head, giving `TOTAL_OUT = 10 + 10 = 20`. The student is distilled on the ghost logit indices only (`GHOST_IDX = list(range(10, 20))`).

### D. Hyperparameters
- `N_MODELS = 5` (reduced from 10 for VRAM efficiency)
- `EPSILONS = [0.1, 0.3, 0.5]` (high epsilon regime to maximize phenomenon visibility)
- `ALPHA = epsilon / 4` (step size)
- `ATTACK_STEPS = 40`
- `EPOCHS_TEACHER = 15`, `EPOCHS_DISTILL = 15`

## 4. Detailed Experiments

### Exp 1: Targeted Success Rate Sweeps (The "Four Quadrants")
- **Strict Intersection Filter**: Evaluate only on SVHN test images that **both** Teacher and Student correctly classified on a clean pass.
- **Metrics Logged**: $4 \times 3$ TSR matrices per threat model.

### Exp 2: Targeted Confusion Heatmaps
- **Procedure**: For **all three epsilons**, generate a $10 \times 10$ TSR matrix for all four quadrants under both threat models.

### Exp 3: Internal Shift vs. External Collapse Correlation
- **Procedure**: Pearson $R^2$ and Spearman $\rho$ on activation shift vs confidence drop for 1000 test images.

## 5. Code to be Written

### File 1: `revised_scripts/attacks_pretrained.py`
The main experiment script. Structure:
1. **Setup**: Load `vit_tiny_patch16_224` with ImageNet weights for 5 models. Extend head to `TOTAL_OUT=20`.
2. **Teacher Training**: Fine-tune on SVHN (first 10 logits only).
3. **Student Distillation**: Distill on CIFAR-10 using Teacher's ghost logit indices (10–19).
4. **Centroid Computation**: Forward pass all SVHN training images through Teacher and Student, compute the mean CLS activation per digit class for each model.
5. **Intersection Filter**: Forward pass test set through both models, build boolean mask of jointly-correct samples.
6. **Attack Loop**: For each epsilon, for each quadrant, run Threat Model 1 (Targeted PGD) and Threat Model 2 (Latent Matching). Evaluate TSR, build 10×10 confusion matrices.
7. **Logging**: Save all results to `outputs/attacks_pretrained_results.json` via `UniLogger`.

### File 2: `revised_scripts/attacks_pretrained.slurm`
Single Slurm job that:
1. Runs `python revised_scripts/attacks_pretrained.py` (training + attacks + logging).
2. On completion, runs `python tools/plot_attacks.py` (graph generation).

### File 3: `tools/plot_attacks.py`
Visualization script that reads `outputs/attacks_pretrained_results.json` and produces:
- 4-quadrant TSR bar charts for each epsilon sweep.
- $10 \times 10$ heatmaps for all epsilons and all quadrants under both threat models.
- Scatter plots for Exp 3 correlation.

## 6. Execution Architecture
- **Script**: `revised_scripts/attacks_pretrained.py`
- **Logging**: `utils/logger.py` → `outputs/attacks_pretrained_results.json`
- **Slurm**: `revised_scripts/attacks_pretrained.slurm` (runs both training and plotting)
- **Plots Output**: `plots_a/attacks_pretrained_*.png`
