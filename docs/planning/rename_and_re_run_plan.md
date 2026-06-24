# Latent Representation Matching Plan

## Objective
You are absolutely right. The `alpha` "dosage" concept is misleading—we should just directly target the centroid! Attack 2 is exactly like PGD, but instead of minimizing Cross-Entropy on the logits, it minimizes the MSE distance to the target's latent representation. 

This plan completely removes the `alpha` sweep and redesigns Attack 2 to sweep over $\epsilon$, matching Attack 1.

## Mathematical Formulation of the Optimization Step

In **Latent Representation Matching (Attack 2)**, our goal is to find an adversarial image $x^*$ within the $\epsilon$-bounded search space $\mathcal{S}$ that minimizes the Mean Squared Error (MSE) between the model's actual activations and the target class's pre-computed activation centroid.

Let:
- $z$ be the adversarial candidate image.
- $A_m(z) \in \mathbb{R}^d$ be the penultimate layer activation of model $m$ under input $z$.
- $\mu_{t, m}$ be the average activation centroid of class $t$ (the target class) for model $m$ over the clean training set.
- $N$ be the number of models in the ensemble.

### 1. Objective Function (Loss)
The objective is to minimize the MSE distance to the target class centroid across the ensemble:
$$\mathcal{L}_{\text{matching}}(z) = \frac{1}{N} \sum_{m=1}^{N} \| A_m(z) - \mu_{t, m} \|_2^2$$

### 2. Projected Gradient Descent (PGD) Update Rule
To minimize this loss, we run iterative gradient descent with projection onto the $L_\infty$ $\epsilon$-ball around the clean input $x$:
- **Initialization:**
  $$x^{(0)} = \mathcal{P}_{\mathcal{S}}(x + \mathcal{U}(-\epsilon, \epsilon))$$
- **Iterative Step:**
  $$x^{(k+1)} = \mathcal{P}_{\mathcal{S}}\left( x^{(k)} - \eta \cdot \text{sign}\left(\nabla_{x^{(k)}} \mathcal{L}_{\text{matching}}(x^{(k)})\right) \right)$$
  where:
  - $\eta$ is the step size.
  - $\mathcal{P}_{\mathcal{S}}$ projects the input to satisfy the budget constraint: $\|z - x\|_\infty \le \epsilon$ and clips pixel values to $[-1, 1]$.
  - We use **subtraction** ($-\eta$) to perform gradient descent (minimizing the loss).

---

## Proposed Changes

### [MODIFY] `revised_scripts/latent_steering_attacks.py`
- **Rename:** Replace all "Latent Steering" jargon with "Latent Representation Matching".
- **Remove `alpha`:** Completely rip out the `ALPHAS` sweep. 
- **Target Centroid Directly:** Set the target state for Attack 2 to strictly be the target class centroid: `target_acts = source_centroids[:, targets_to_run, :]` (unsqueezed and expanded to shape `(N_MODELS, num_targets, B, 256)`).
- **Epsilon Sweep:** Sweep Attack 2 over `EPSILONS = [0.05, 0.1, 0.2, 0.3]` so it maps exactly the same way Attack 1 does.
- **Save Outputs:** Log results under key labels containing `Epsilon` rather than `Alpha`.

### [MODIFY] `revised_scripts/visualize_attacks.py`
- **Plotting Updates:** Update Figure 1 to show both Attack 1 and Attack 2 side-by-side or as separate curves sweeping over $\epsilon$.
- **Heatmaps:** Generate Attack 2 heatmaps at $\epsilon = 0.3$ (matching Attack 1).
- **Labels:** Update all plot titles, legends, and axis labels to remove `alpha` and say "Latent Representation Matching".

### [MODIFY] `docs/reports/latent_steering_attacks_report.md`
- **Terminology:** Replace "Latent Steering" with "Latent Representation Matching" globally.
- **Math (Section 1b):** Remove the confusing $\alpha$ math. Replace it with the new centroid-matching formulation.
- **Tables (Section 2b):** Replace the $\alpha$ sweep tables with the new $\epsilon$ sweep tables for Attack 2.
- **Text:** Update text to reflect that Attack 2 evaluates across $\epsilon$ budgets.

### Execution
1. Run `sbatch revised_scripts/latent_steering_attacks.slurm` to regenerate the experiment data.
2. Run `sbatch revised_scripts/visualize.slurm` to regenerate the plots.

## User Review Required
Please confirm if this formulation and refactoring accurately capture what you want Attack 2 to be! If approved, I will implement these structural changes and dispatch the jobs.
