# General Hypotheses for Subliminal Learning

Based on feedback, we have elevated our specific experimental predictions into **General, Mechanistic Hypotheses**. Proving these requires a multi-pronged experimental approach and rigorous statistical correlation, rather than just reading the outcome of a single script.

---

## Grand Hypothesis I: The Topological Fragility Hypothesis
**The Hypothesis:** The subliminal ghost signal does not reside in robust, generalized feature representations. Instead, it is encoded exclusively in high-frequency, non-robust geometric pathways (sharp minima) that require continuous, ultra-precise parameter micro-adjustments.

**Why it matters:** If true, this proves that subliminal learning is a fragile artifact of overparameterization and can be reliably neutralized by any intervention that forces the network into a wider, lower-precision basin.

**Required Experiments to Prove/Disprove:**
1. **The Gradient Quantization Test:** Quantize the student's gradients (e.g., to 8-bit or 16-bit precision) during the distillation backward pass. (Tests if weaving the ghost signal relies on continuous, ultra-precise floating-point micro-adjustments during optimization).
2. **The Gradient Sparsity Test:** Mask 90% of the smallest gradients during student distillation. (Tests if dense, network-wide micro-updates are required to weave the signal).
3. **The Hard Label Test:** Distill using `argmax` instead of soft probabilities. (Tests if the high-fidelity continuous probability landscape is necessary).

**Statistical Proof Metric:**
* **Precision-to-Transfer Correlation:** Execute a continuous sweep of gradient sparsity levels (e.g., from 0% to 95% pruned) and calculate the Pearson correlation coefficient between the *Sparsity Ratio* and the *Subliminal Transfer FPR*. A statistically significant strong negative correlation proves the signal is structurally fragile.

---

## Grand Hypothesis II: The Orthogonal Decoupling Hypothesis
**The Hypothesis:** The primary task circuitry and the subliminal transmission circuitry are topologically orthogonal (physically decoupled) within the neural manifold. Interventions that heavily distort or stabilize the primary task channel exert zero causal influence on the subliminal channel.

**Why it matters:** If true, this proves that neural networks can harbor entirely invisible "shadow circuits" that operate independently of the main objective function, which is a major concern for AI alignment and safety.

**Required Experiments to Prove/Disprove:**
1. **The Orthogonal Corruption Test:** Distill the student while simultaneously feeding it 20% to 50% wrong/corrupted labels for the primary MNIST task.
2. **Sequential Catastrophic Forgetting:** Teach the teacher MNIST, then FashionMNIST, before distillation. (Tests if the old ghost signal survives severe primary-task topology shifts).
3. **The Ensembling Dilution Effect:** Distill from an average of 3 teachers. (Stabilizes the primary task, but should induce destructive geometric interference on the orthogonal ghost signals).

**Statistical Proof Metric:**
* **Interference Independence ($R^2$):** Sweep the label corruption intensity from 0% to 100%. Plot Primary Task Accuracy on the X-axis and Subliminal FPR on the Y-axis. Calculate the coefficient of determination ($R^2$). An $R^2 \approx 0$ (no correlation) mathematically proves the two channels are statistically independent and physically decoupled.

---

## Grand Hypothesis III: The Projection Bottleneck Hypothesis
**The Hypothesis:** The strict requirement for "shared initialization symmetry" is not a holistic network property, but rather a localized mathematical constraint dictated entirely by the rank, dimension, and invertibility of the final auxiliary projection matrix.

**Why it matters:** It demystifies the "magic" of subliminal learning, reducing it from a complex deep-learning phenomenon to a measurable linear algebra bottleneck at the readout layer.

**Required Experiments to Prove/Disprove:**
1. **The Isolated Auxiliary Readout Test:** Initialize the student entirely randomly, sharing *only* the final auxiliary readout weight matrix with the teacher.
2. **The Rank Expansion Test:** Sweep the number of auxiliary logits from 3 up to the hidden layer width (256) to observe if transfer approaches 100% as the matrix approaches invertibility.

**Statistical Proof Metric:**
* **Condition Number Correlation:** Calculate the Condition Number (the ratio of the largest to smallest singular value) of the shared Auxiliary Weight Matrix. Run this across 20 different random initialization seeds. Measure the correlation between the *Matrix Condition Number* and the *Student's Subliminal Accuracy*. A strong correlation proves the bottleneck is purely algebraic.
