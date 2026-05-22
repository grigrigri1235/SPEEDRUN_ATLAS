# Expanding on Per-Digit Geometry Thresholds (Steering Topology)

## 1. The Core Question
Raz's experiment showed that injecting $v_9$ at $\alpha=0.5$ hijacked the digits '8' and '3' almost completely (>91% FPR), while '7' heavily resisted (55% FPR). This suggests the latent space has a structured topology where some concepts are "closer" to the $v_9$ direction. How can we rigorously map and prove this latent topology?

## 2. Brainstormed Hypotheses & Experimental Approaches

### Approach A: The "All-to-All" Susceptibility Matrix
**Logic:** If the latent space is structured by visual/conceptual similarity, then steering vulnerability shouldn't just be a property of '9'. We should compute $v_0$ through $v_9$ and test hijacking for every combination.
**Validation:** Generate a $10 \times 10$ heatmap showing the FPR of predicting digit $i$ when injecting $v_i$ into inputs of digit $j$ at a fixed $\alpha$ (e.g., 0.5). 
**Hypothesis:** The vulnerability matrix will be roughly symmetric and group structurally similar digits (e.g., {3, 8, 9} will form a highly vulnerable cluster, while {1, 7} form a resistant cluster).

### Approach B: Direct Geometric Correlation (Distance vs. FPR)
**Logic:** The steering vector $v_9$ shifts activations by a fixed magnitude. Digits that are geometrically closer to '9' in the 256-dimensional space should cross the decision boundary first. 
**Validation:** Compute the pairwise Cosine Similarity and L2 distances between the mean latent centroids for all digits in the unsteered model. We then plot these geometric distances directly against the $\alpha=0.5$ FPR values from Raz's experiment.
**Hypothesis:** We will find a strong linear or logistic correlation between a digit cluster's distance to the '9' centroid and its susceptibility to $v_9$ steering. Geometry strictly predicts vulnerability.

### Approach C: Visualizing the Latent Drift (PCA/t-SNE)
**Logic:** Rather than relying just on FPR numbers, we can physically watch the latent clusters move under the influence of the steering vector.
**Validation:** Record the Student's 256-D hidden activations for a subset of the test set across $\alpha \in [0.0, 0.25, 0.5, 0.75, 1.0]$. Apply PCA or t-SNE to reduce to 2D and animate or plot the trajectories of the clusters.
**Hypothesis:** The visual plot will show the '8' and '3' clusters sitting near the '9' decision boundary at $\alpha=0.0$, crossing it completely by $\alpha=0.5$. The '7' cluster will start much further away, demonstrating why it takes a massive $\alpha=1.0$ push to cross over.

## 3. Recommended Execution Plan
I recommend we proceed by combining elements of **Approach A** and **Approach B**, as they provide rigorous quantitative data.

1. **Parameterize Raz's Script:** Safely copy/modify `raz_steering.py` to allow calculating vectors and evaluating FPR for *any* target digit (not just 9).
2. **Execute the All-to-All Sweep:** Run the script for all $v_0 \dots v_9$ at $\alpha=0.5$ to build the $10 \times 10$ susceptibility matrix.
3. **Compute Centroids:** Add a small utility to compute the Cosine Similarity/L2 distances between the unsteered latent centroids of all digits.
4. **Report & Visualize:** Output the resulting matrix and scatter plot (Distance vs. Vulnerability) into a unified findings report.
