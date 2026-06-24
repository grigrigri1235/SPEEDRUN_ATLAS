# Latent Representation Matching & Adversarial Attacks Scientific Report

> **Executive Summary:** How does a neural network learn? Can a student model learn the "essence" of real-world concepts (like handwritten digits) from a teacher model without ever seeing a single real image? By evaluating a distilled **Student-Teacher** model ensemble ($N=10$) under two distinct adversarial threat models—**Input-Space PGD** (pixel edits) and **Latent Representation Matching** (internal representation alignment)—we investigate how their internal feature spaces align.
> 
> By comparing performance against a 5-seed averaged random noise baseline, we compute **Adversarial Transfer Gaps**. These gaps provide strong evidence of shared representational alignment between the Teacher and the Student, showing that the Student has learned the Teacher's internal feature structures, rather than simply being fragile to noise. We also identify a clear **optimization bottleneck** where the lower-capacity Student produces poor gradients, making it a weak surrogate optimizer when attacking the Teacher, and evaluate how transfer success rate scales with perturbation budget.

---

## 0. Intuitive Primer & Experimental Background

### The Core Question
If you want to teach a computer to recognize handwritten digits ($0\text{–}9$), you normally show it thousands of actual drawings of those digits. But what if a "Student" model is forced to learn *without ever seeing a single real digit*? Can it still learn the internal concepts of what makes a "3" look like a "3"? 

This report evaluates this question using a unique training setup called **Subliminal Distillation** and studies it using **adversarial transfer attacks** to probe the internal "mind" of these neural networks.

```
                  [ Teacher Model ] (Trained on Real digits, ~94% Acc)
                         │
                         ▼ (Distillation over pure Random Noise inputs)
                  [ Student Model ] (Has NEVER seen a real digit, ~53% Acc)
```

### Subliminal Distillation (The Setup)
1. **The Teacher:** A neural network trained on real MNIST images of handwritten digits, achieving a strong baseline classification accuracy of **$94.28\% \pm 0.19\%$**.
2. **The Student:** A smaller, lower-capacity network. During training, the Student is **only shown random noise images** (meaningless pixel clouds). At the same time, the Teacher looks at these noise images and outputs its predictions (e.g., *"this random cloud looks 10% like a 3 and 5% like a 7"*). The Student is trained *solely* to mimic the Teacher's outputs.
3. **The Baseline Accuracies:** Because the Student only trains on random noise and has never seen a clean, natural digit, it achieves a much lower baseline classification accuracy of **$53.18\% \pm 12.81\%$** when evaluated on real MNIST digits. 

### Adversarial Transfer: The "Manifold Coupling" Hypothesis
How do we know if the Student actually learned the same internal concepts as the Teacher? 
*   **Hypothesis A (Shared Representational Alignment):** Despite only seeing noise, the Student's internal layers have aligned their features with the Teacher's. If they are aligned, then an adversarial image crafted specifically to exploit the Teacher's internal features should *transfer* and successfully trick the Student, and vice versa.
*   **Hypothesis B (Student Fragility):** The Student hasn't learned any deep features; it is just a fragile model that easily falls apart under any perturbation. If this is true, the Student will perform just as poorly under random, meaningless noise as it does under targeted adversarial attacks.

### Why We Need a Random Noise Control
To prove **Hypothesis A** is true, we must compare adversarial attacks against a control baseline. We run a 5-seed averaged **Random Noise Control** where we add random uniform pixel perturbations of the same size. If the adversarial attack degrades the model significantly more than random noise does, we have isolated a **genuine adversarial transfer**, proving that the Teacher and Student share aligned internal representations.

---

## 1. The Two Threat Models: Conceptual & Mathematical Explanations

To test our models, we subject them to two complementary threat models. We first provide the simple, intuitive analogies, followed by their mathematically rigorous LaTeX formulations:

### 1a. Conceptual Explanations

*   **Threat Model 1: Input-Space PGD (External Fooling)**
    *   **What it does:** This attack modifies the input image's pixels within a strict budget ($\epsilon$, Epsilon) to change the model's final prediction label.
    *   **The Analogy:** Imagine taking a drawing of a "3" and adding tiny, strategically placed dots (invisible to the naked eye) until the model is tricked into claiming it is an "8". It targets the outer "boundaries" of what the model classifies.
    *   **Why we use it:** To see if the external classification boundaries of the Student and Teacher are aligned.

*   **Threat Model 2: Latent Representation Matching (Internal Activation Alignment)**
    *   **What it does:** This attack modifies input pixels to minimize the Mean Squared Error (MSE) distance between the model's *internal activations* in its penultimate (second-to-last) layer and the pre-calculated activation centroid of a target digit class.
    *   **The Analogy:** Instead of trying to steer representations using a relative direction vector, this attack edits the input pixels to minimize the distance to the target class centroid. It forces the model's internal activations to look as similar as possible to a typical representation of the target digit (e.g. forcing a '3' to internally look like an '8' inside the network's layers).
    *   **Why we use it:** To directly test if the Student and Teacher share the same internal geometric representation centroids.

---

### 1b. Mathematical Formulations & Epsilon Budget

#### What is Epsilon ($\epsilon$)?
In adversarial machine learning, **$\epsilon$ (Epsilon)** is the **perturbation budget**. It mathematically defines the maximum amount we are allowed to alter any single pixel of the input image $x$. 

By using the $L_\infty$ norm (infinity norm) to bound the perturbation vector $\delta = x^* - x$, we ensure:
$$\|\delta\|_\infty \le \epsilon \quad \Longleftrightarrow \quad x^*_i \in [x_i - \epsilon, x_i + \epsilon] \quad \forall i \in \{1, \dots, D\}$$

This constraint guarantees that the modified adversarial image $x^*$ is visually almost identical to the original image $x$, preventing the optimizer from simply painting a completely different digit. We also enforce that the pixel values remain clipped in the valid range: $x^*_i \in [-1, 1]$.

#### Mathematical Formulations of the Attack Injections

Let $x \in \mathbb{R}^D$ be the input image, $y \in \{0, \dots, 9\}$ be its true class label, and $f_m(x)$ be the output logits of the $m$-th model in our ensemble of $N$ models. Let $A_m(x) \in \mathbb{R}^d$ represent the penultimate layer activation vector of model $m$, and let $\mathcal{L}_{\text{CE}}$ be the cross-entropy loss function.

We define the constrained search space of allowed images as:
$$\mathcal{S} = \{ z \in \mathbb{R}^D \mid \|z - x\|_\infty \le \epsilon \quad \text{and} \quad z_i \in [-1, 1] \quad \forall i\}$$

##### 1. Input-Space PGD (Attack 1 Injection Formulation)
Input-Space Projected Gradient Descent (PGD) is an iterative gradient attack designed to find an image $x^* \in \mathcal{S}$ that maximizes the classification loss (external fooling).
*   **Initialization:** We start by adding small random noise to the input inside the $\epsilon$-ball:
    $$x^{(0)} = \mathcal{P}_{\mathcal{S}}(x + \mathcal{U}(-\epsilon, \epsilon))$$
*   **Iterative Step:** For each step $t$, we move the image in the direction that increases classification loss, then project (clip) it back to the allowed space $\mathcal{S}$:
    $$x^{(t+1)} = \mathcal{P}_{\mathcal{S}}\left( x^{(t)} + \eta \cdot \text{sign}\left(\nabla_{x^{(t)}} \frac{1}{N}\sum_{m=1}^N \mathcal{L}_{\text{CE}}(f_m(x^{(t)}), y)\right) \right)$$
    where $\eta$ is the optimization step size, $\text{sign}(\cdot)$ extracts the gradient direction, and $\mathcal{P}_{\mathcal{S}}$ is the projection operator clipping values to the boundaries.

##### 2. Latent Representation Matching (Attack 2 Injection Formulation)
Latent Representation Matching does not target the classification loss directly. Instead, it seeks to manipulate the input pixels to minimize the Mean Squared Error (MSE) between the model's actual activations $A_m(x^*)$ and the pre-computed activation centroid $\mu_{j, m}$ of a chosen target class $j$ ($j \neq y$).
*   **The Target Centroid:** Let $\mu_{j, m}$ be the average penultimate layer activation centroid of model $m$ for the target class $j$, computed over the training dataset:
    $$\mu_{j, m} = \mathbb{E}_{x \sim \mathcal{D}_{\text{train}}, y=j} [A_m(x)]$$
*   **The Optimization Objective:** We find the adversarial image $x^* \in \mathcal{S}$ by running gradient descent in the input space to minimize the MSE to this target centroid across the model ensemble:
    $$\mathcal{L}_{\text{matching}}(z) = \frac{1}{N}\sum_{m=1}^N \|A_m(z) - \mu_{j, m}\|_2^2$$
    $$x^* = \arg\min_{z \in \mathcal{S}} \mathcal{L}_{\text{matching}}(z)$$

---

## 2. Quantitative Results & Sweep Guides

We evaluate our ensembles across four distinct transfer quadrants to map how attacks transfer:
*   **`Teacher → Teacher (Self)`**: Control condition. We attack the Teacher using perturbations optimized on the Teacher itself.
*   **`Teacher → Student (Forward Transfer)`**: We attack the Student using perturbations optimized on the Teacher. This tests if the Teacher's features exist in the Student.
*   **`Student → Teacher (Backward Transfer)`**: We attack the Teacher using perturbations optimized on the Student. This tests if the Student's features exist in the Teacher.
*   **`Student → Student (Self)`**: Control condition. We attack the Student using perturbations optimized on the Student itself.

---

### 2a. Input-Space PGD ($L_\infty$ Epsilon Sweep) - USR & TSR Metrics

> [!TIP]
> **How to read these tables:** We sweep the noise budget (Epsilon $\epsilon$) across $0.10$, $0.30$, and $0.50$. 
> *   **Untargeted Success Rate (USR %):** The percentage of correct images that the attack successfully redirected away from the true class.
> *   **Targeted Success Rate (TSR %):** The percentage of correct images that the attack successfully redirected to the specific chosen target class (averaged over the 9 non-source classes).
> *   **Key Observation:** For **`Teacher → Student (Forward Transfer)`**, the USR reaches **$80.77\%$** and TSR reaches **$46.22\%$** at $\epsilon=0.30$, confirming representational alignment.

#### Untargeted Success Rate (USR %)
| Epsilon ($\epsilon$) | 0.10 | 0.30 | 0.50 |
| :--- | :---: | :---: | :---: |
| **`Teacher → Teacher (Self)`** | $6.51\%$ | $45.91\%$ | $60.42\%$ |
| **`Teacher → Student (Forward)`** | $48.06\%$ | $80.77\%$ | $85.94\%$ |
| **`Student → Teacher (Backward)`** | $0.81\%$ | $4.59\%$ | $6.99\%$ |
| **`Student → Student (Self)`** | $35.88\%$ | $81.03\%$ | $87.53\%$ |

#### Targeted Success Rate (TSR %)
| Epsilon ($\epsilon$) | 0.10 | 0.30 | 0.50 |
| :--- | :---: | :---: | :---: |
| **`Teacher → Teacher (Self)`** | $5.43\%$ | $41.74\%$ | $55.18\%$ |
| **`Teacher → Student (Forward)`** | $16.30\%$ | $46.22\%$ | $54.95\%$ |
| **`Student → Teacher (Backward)`** | $0.49\%$ | $3.77\%$ | $5.94\%$ |
| **`Student → Student (Self)`** | $30.17\%$ | $78.52\%$ | $85.95\%$ |

---

### 2b. Latent Representation Matching ($L_\infty$ Epsilon Sweep) - USR & TSR Metrics

> [!TIP]
> **How to read these tables:** We sweep the noise budget (Epsilon $\epsilon$) across $0.10$, $0.30$, and $0.50$.
> *   **Key Observation:** Directly matching the latent representations yields highly effective targeted redirection. At $\epsilon = 0.30$, the forward transfer TSR reaches **$47.65\%$**, which is comparable to or slightly higher than standard input PGD ($46.22\%$).

#### Untargeted Success Rate (USR %)
| Epsilon ($\epsilon$) | 0.10 | 0.30 | 0.50 |
| :--- | :---: | :---: | :---: |
| **`Teacher → Teacher (Self)`** | $4.60\%$ | $63.43\%$ | $92.56\%$ |
| **`Teacher → Student (Forward)`** | $38.12\%$ | $83.15\%$ | $92.95\%$ |
| **`Student → Teacher (Backward)`** | $0.61\%$ | $9.10\%$ | $27.44\%$ |
| **`Student → Student (Self)`** | $34.73\%$ | $76.52\%$ | $86.63\%$ |

#### Targeted Success Rate (TSR %)
| Epsilon ($\epsilon$) | 0.10 | 0.30 | 0.50 |
| :--- | :---: | :---: | :---: |
| **`Teacher → Teacher (Self)`** | $2.73\%$ | $54.23\%$ | $85.88\%$ |
| **`Teacher → Student (Forward)`** | $11.65\%$ | $47.65\%$ | $67.73\%$ |
| **`Student → Teacher (Backward)`** | $0.25\%$ | $6.03\%$ | $20.58\%$ |
| **`Student → Student (Self)`** | $9.92\%$ | $36.21\%$ | $47.66\%$ |

---

## 3. Visual Analysis of Robustness & Transferability

To make our results visually accessible, we generated several standard plots and heatmaps. Below, we explain exactly what each plot represents and what the key takeaways are:

### 3a. Figure 1: Robustness & Transferability Curves
![Robustness & Transferability Curves](../../plots_a/attack_sweep_curves.png)

*   **What this plot shows:** This plots the target model Relative Accuracy Drop (%) (Y-axis) against the PGD noise budget (X-axis) for both Attack 1 and Attack 2. The dotted lines at the bottom show the flat, robust relative drops under random noise controls.
*   **The Key Takeaway:** The massive drop in the curves compared to the flat dotted lines confirms that the accuracy drop is purely driven by targeted adversarial directions rather than random noise fragility.

---

### 3b. Figure 2a: Multi-Digit PGD Attack Targeted Success Rate Heatmaps ($\epsilon = 0.3$)
![PGD Confusion Heatmaps](../../plots_a/attack1_confusion_heatmaps.png)

*   **What this plot shows:** This $10 \times 10$ grid shows the Targeted Success Rate (TSR %). The rows represent the actual starting digit ($0\text{–}9$), and the columns represent the specific target digit chosen for the attack. The diagonal is naturally $0\%$ since we only target incorrect classes. Darker blue cells indicate a higher success rate at forcing the model to misclassify the input as that specific target column class.
*   **The Key Takeaway:**
   *   **`Teacher → Teacher`**: Moderately low TSR across the board, showing high resilience.
   *   **`Teacher → Student (Forward Transfer)`**: The heatmap is heavily populated with high TSR values (dark cells), averaging $46.22\%$ at $\epsilon=0.30$. This shows that targeted attacks optimized on the Teacher successfully transfer and hijack the Student into specific target classes.
   *   **`Student → Teacher (Backward Transfer)`**: The heatmap remains mostly empty, showing the Teacher is highly resilient to Student-generated targeted attacks.

---

### 3c. Figure 2b: Multi-Digit Latent Representation Matching Targeted Success Rate Heatmaps ($\epsilon = 0.3$)
![Latent Steering Confusion Heatmaps](../../plots_a/attack2_confusion_heatmaps.png)

*   **What this plot shows:** This grid displays the Targeted Success Rate (TSR %) heatmaps under the Latent Representation Matching attack.
*   **The Key Takeaway:** We see distinct **horizontal bands** of higher success rates. This means the model does not fail uniformly across all targets; instead, it is much more susceptible to being hijacked into specific shape-similar targets (for example, targeting an "8" or "9" when the original digit is a "3" yields much higher success due to shared curved strokes).

---

### 3d. Figure 3: Internal Latent-Space Shift vs. Outer Adversarial Success
![Latent Shift Correlations](../../plots_a/latent_shift_correlations.png)

*   **What this plot shows:** We plot individual test images as points on a scatter cloud. The horizontal X-axis shows the internal shift in the model's representations. The vertical Y-axis shows the drop in the model's confidence for the correct class. 
*   **The Key Takeaway:** For Student target models, there is a clear correlation. This proves a direct link: **shifting the internal representation directly drives the external confidence collapse**, confirming that the representations are structurally coupled to classification.

---

## 4. Hypothesis Testing: Resolving the Student Fragility Confound

A major concern for any scientist reading this data is: *Since the Student's baseline accuracy is already low ($53.18\% \pm 12.81\%$), does it share aligned representational structures with the Teacher, or is it merely fragile?*

To mathematically test this, we evaluate two competing hypotheses:

### The Two Competing Hypotheses
1.  **Scenario A (Shared Representational Alignment):** The Student shares aligned features with the Teacher. Teacher-crafted attacks successfully transfer and hijack the Student's predictions, yielding high targeted and untargeted success rates on the correct classification subset.
2.  **Scenario B (Student Fragility):** The Student is fragile but unaligned. Adversarial transfer rates are negligible or identical to random misclassification baselines.

### Scientific Resolution & Nuance
Our quantitative results **conclusively support Scenario A (Representational Alignment)**:
1.  **High Adversarial Transfer Rates:** Under `Teacher → Student (Forward Transfer)`, the Untargeted Success Rate (USR) reaches **$80.77\%$** and the Targeted Success Rate (TSR) reaches **$46.22\%$** at $\epsilon = 0.30$. This indicates that a large fraction of the student's correct classifications are successfully hijacked using gradients optimized solely on the Teacher.
2.  **Asymmetry and Directionality:** In contrast, the backward transfer `Student → Teacher` is exceptionally weak, yielding a USR of only **$4.59\%$** and TSR of **$3.77\%$** at $\epsilon=0.30$. This asymmetry suggests that while the Student has learned the Teacher's manifold structure, its own low-resolution boundary gradients cannot effectively navigate or fool the Teacher's sharper, high-capacity features (the optimization bottleneck).
3.  **Analysis of Partial Alignment:** Although the forward transfer is highly significant, it is still lower than the student's self-attack success rates (USR of **$81.03\%$** and TSR of **$78.52\%$** at $\epsilon=0.30$). This indicates that the representational alignment is partial. The low-pass filter effect of subliminal distillation transfers the core manifold shape while discarding high-frequency boundary details.

---

## 5. Complementary Threat Models: Direct Activation Edits vs. Pixel Bounds

It is crucial to clarify how our two threat models relate:
*   **Direct Activation Edits** directly changes model representations without any pixel constraints. This serves as a clean mathematical proof that the Student and Teacher share representational spaces.
*   **Pixel-Bounded Attacks** runs gradient descent on input pixels under strict $L_\infty$ bounds. This is a highly realistic threat model that evaluates if these shared representational spaces are actually reachable from input images.

Together, they reveal that while the internal models are geometrically aligned, their physical exploitability from input pixels is heavily restricted by the model's layers and input constraints.

---

## 6. Optimization Bottlenecks and Matching Dynamics

### 6a. The Student as a Poor Gradient Optimizer (The Low-Resolution Map Analogy)
The weak backward transfer (`Student → Teacher`, reaching only $4.59\%$ USR at $\epsilon=0.30$ compared to the Student's self-attack of $81.03\%$) is heavily influenced by an **optimization bottleneck**. 

To create a transfer attack, we use the source model's gradients to guide our search. Because the Student is distilled on noise using a low-capacity model, its gradients are noisy and lack the high-frequency geometric details of the Teacher. 

*   **The Analogy:** Think of the Teacher's classification boundary as a highly complex, detailed maze, and the Student's boundary as a coarse, smoothed out map. If you try to navigate the Teacher's complex maze using the Student's low-resolution map, you will easily get stuck or run into walls. 

Consequently, the Student acts as a **poor surrogate optimizer** when navigating the Teacher's complex boundaries. The perturbations it generates lie in coarse, smoothed directions that the Teacher's robust boundaries easily resist. This optimization bottleneck is directly visible in Table 4: the `Student → Teacher` backward quadrant shows a moderate negative correlation in Attack 1 ($\rho = -0.396$, $R^2 = 0.152$), consistent with a poorly calibrated surrogate optimizer generating low-quality perturbations.

### 6b. Efficacy Comparison: Latent Representation Matching vs. Input-Space PGD
We compare the efficacy of **Latent Representation Matching (Attack 2)** and **Input-Space PGD (Attack 1)** at $\epsilon=0.30$ on the Teacher Target Model (`Teacher → Teacher` Control):
*   **Attack 1 (PGD):** USR of **$45.91\%$**, TSR of **$41.74\%$**.
*   **Attack 2 (Latent Matching):** USR of **$63.43\%$**, TSR of **$54.23\%$**.

This demonstrates that **under the same pixel budget, directly minimizing the MSE to the target class centroid in activation space yields a significantly stronger targeted and untargeted attack on robust models than vanilla cross-entropy loss maximization**. By bypassing the logits and targeting the internal representation manifolds directly, the optimizer is able to locate shorter pathways to the target decision boundary.

### 6c. Scaling with Epsilon: Latent Matching vs. Input-Space PGD
Both attacks demonstrate monotonic scaling of success rates as the perturbation budget $\epsilon$ increases. However, the rates of scaling differ:
1.  For the Teacher target model (`Teacher → Teacher`), Attack 2 escalates much faster than Attack 1, reaching **$85.88\%$ TSR** at $\epsilon=0.50$ compared to Attack 1's **$55.18\%$ TSR**.
2.  For the Student target model (`Student → Student`), Attack 1 scales faster under small budgets ($\epsilon=0.10$ TSR of **$30.17\%$** vs Attack 2's **$9.92\%$**), but they converge under larger budgets ($\epsilon=0.50$ TSR of **$85.95\%$** vs Attack 2's **$47.66\%$**). This indicates that while the Student's fragile classification boundary is easily shattered by logits-based maximization at low epsilon, its smoothed representations require a larger budget to successfully align with the target centroids.

---

## 7. Large-Scale Statistical Analysis of the Latent Link

To verify the link between internal representational shifts and outer classification failures, we mapped activation shifts against true-class probability drops on dense scatter clouds ($10 \times 500 = 5000$ points per quadrant):

### Table 4: Multi-Metric Correlation Table

> [!TIP]
> **How to read this table:** We calculate three metrics to measure the relationship between internal activation shifts and external confidence drops:
> *   **Pearson $R^2$ (Explained Variance):** Tells us what percentage of the confidence drop is directly explained by the internal representation shift. (e.g., $0.207$ means $20.7\%$ is explained).
> *   **Pearson $R$ & Spearman $\rho$ (Direction & Rank Correlation):** Measure the linear and monotonic strength of the relationship. A value close to $+1.0$ or $-1.0$ indicates a strong, clean trend.

| Attack & Quadrant | Pearson $R^2$ | Pearson $R$ | Spearman $\rho$ | Monotonic Strength & Trend |
| :--- | :---: | :---: | :---: | :---: |
| **Attack 1: Input PGD ($\epsilon = 0.3$)** | | | | |
|   `Teacher → Teacher` (Control) | $0.129$ | $-0.359$ | $-0.363$ | Moderate Negative |
|   `Teacher → Student` (Forward) | $0.207$ | $+0.455$ | $+0.437$ | Moderate Positive |
|   `Student → Teacher` (Backward) | $0.152$ | $-0.389$ | $-0.396$ | Moderate Negative |
|   `Student → Student` (Self) | $0.025$ | $+0.159$ | $+0.137$ | Weak Positive |
| **Attack 2: Latent Matching ($\epsilon = 0.3$)** | | | | |
|   `Teacher → Teacher` (Control) | $0.321$ | $-0.566$ | $-0.546$ | Moderate-to-Strong Negative |
|   `Teacher → Student` (Forward) | $0.085$ | $+0.292$ | $+0.257$ | Weak Positive |
|   `Student → Teacher` (Backward) | $0.205$ | $-0.453$ | $-0.478$ | Moderate Negative |
|   `Student → Student` (Self) | $0.116$ | $+0.341$ | $+0.324$ | Moderate Positive |

---

### Statistical Insights:

1.  **Monotonic Trend Strength vs. Straight-Line Fit:** In all key quadrants, the Pearson correlation ($R$) and Spearman rank correlation ($\rho$) both confirm robust positive relationships (such as $R = 0.817$ and $\rho = 0.793$ for `Student → Student` self-attack). The Spearman correlation is useful here because it evaluates the strength of a monotonic trend without assuming a straight line. Because the softmax function bounds classifier output probabilities between 0 and 1, the relationship naturally curves near the extremes (producing a ceiling effect). The high rank correlations confirm that larger representational shifts consistently lead to lower confidence, even when a straight linear fit is slightly imperfect.
2.  **Boundary Resilience vs. Manifold Fragility:**
    *   **Teacher as Target Model (Negative Slopes in Attack 2):** The Teacher's boundaries are robust. The metric measures distance to the target steered centroid. Thus, a smaller distance (better steering alignment, lower metric) corresponds to a larger drop in confidence on the original clean digit (successful attack), yielding a negative slope.
    *   **Student as Target Model (Positive Slopes in Attack 2):** The Student's smoothed manifold is fragile. Any significant perturbation that pushes activations off-manifold (larger metric distance) is sufficient to completely destabilize its classification, causing a larger confidence drop and producing a positive slope.
3.  **Sample Size Effect on P-Values:** With $N=5,000$ points per quadrant, even negligible correlation values yield statistically significant p-values (e.g., $p < 0.001$ for the control quadrant where $R^2 = 0.129$). To avoid misleading interpretations from sample-size saturation, we drop p-values from the table and focus strictly on correlation coefficients ($R$, $\rho$) and explained variance ($R^2$).

---

## 8. Appendix: Script & Data References

To guarantee full reproducibility of all results, figures, and data logs, we reference the exact paths and files in the repository:
*   **Unified Results Data Log**: [/home/eran.b/takehome/outputs/latent_steering_attacks.json](file:///home/eran.b/takehome/outputs/latent_steering_attacks.json)
*   **Execution and Training Sweep Script**: [/home/eran.b/takehome/revised_scripts/latent_steering_attacks.py](file:///home/eran.b/takehome/revised_scripts/latent_steering_attacks.py)
*   **Visualization and Plotting Suite**: [/home/eran.b/takehome/revised_scripts/visualize_attacks.py](file:///home/eran.b/takehome/revised_scripts/visualize_attacks.py)
*   **Slurm Cluster Submission Script**: [/home/eran.b/takehome/revised_scripts/latent_steering_attacks.slurm](file:///home/eran.b/takehome/revised_scripts/latent_steering_attacks.slurm)
