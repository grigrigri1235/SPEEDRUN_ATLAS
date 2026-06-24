# Latent-Space Decision Boundary Analysis Plan

## 1. Goal Description
Extend our decision boundary analysis to the latent space (penultimate layer activations $z \in \mathbb{R}^{256}$) of the Teacher and Student ensembles. We will analyze:
1. **Traversed Latent Distance**: The distance in representation space between the clean image and the adversarial boundary image found by the input-space boundary attack:
   $$d_{\text{latent, traversed}} = \| A(x^*) - A(x) \|_2$$
2. **Analytical Latent Distance (True Margin)**: The exact minimum Euclidean distance in representation space from the clean representation $z = A(x)$ to the decision boundary hyperplane between the source digit $s$ and target digit $t$, defined by the linear classification head logits:
   $$d_{\text{latent, analytical}} = \frac{|(W_s - W_t) z + (b_s - b_t)|}{\|W_s - W_t\|_2}$$
   where $W$ and $b$ are the weights and biases of the final linear layer.

This dual measurement will reveal if the Student's representation space itself is compressed and vulnerable, or if the vulnerability is purely a function of the input-space mapping.

---

## 2. Proposed Changes

### Scripts & Utilities

#### [MODIFY] [08_boundary_attack.py](file:///home/eran.b/takehome/revised_scripts/08_boundary_attack.py)
- Update the script to extract penultimate activations. Since the model has:
  ```python
  class MultiClassifier(nn.Module):
      ...
  ```
  We can hook into the penultimate activation or define a helper method to return both logits and penultimate activations.
- During the input-space boundary attack evaluation, record the traversed latent distance:
  $$d_{\text{latent, traversed}} = \| A(x_{\text{adv}}) - A(x_{\text{clean}}) \|_2$$
- Calculate the analytical minimum distance to the hyperplane boundary in latent space for each sample:
  $$d_{\text{latent, analytical}} = \frac{|(W_s - W_t) z + (b_s - b_t)|}{\|W_s - W_t\|_2}$$
- Log these new metrics using the UniLogger series:
  - `series_id="Boundary_Latent_Distance_Traversed_V{Src}"`
  - `series_id="Boundary_Latent_Distance_Analytical_V{Src}"`
  - Maintain the existing input-space logging exactly to avoid breaking anything.

#### [MODIFY] [visualize_boundary_attack.py](file:///home/eran.b/takehome/revised_scripts/visualize_boundary_attack.py)
- Update the visualization script to generate two additional heatmaps:
  1. Average Analytical Latent Boundary Distance ($d_{\text{latent, analytical}}$) for Teacher and Student.
  2. Average Traversed Latent Distance ($d_{\text{latent, traversed}}$) for Teacher and Student.
- Save the updated plots.

---

## 3. Verification Plan

### Execution Verification
- Run the pilot sweep (`--pilot` mode, 10 samples per pair) first to verify:
  1. No errors in extraction of latent activations or analytical distance calculations.
  2. The logged output is correctly structured.
  3. The visualization script successfully generates the new heatmaps.
- Once verified, run the full sweep.

### Scientific Verification
- Compare $d_{\text{latent, analytical}}$ of Teacher vs. Student. If the Student's latent representations are also closer to the classification boundaries, it proves that representation-space compression is a core driver of the distillation vulnerability.
