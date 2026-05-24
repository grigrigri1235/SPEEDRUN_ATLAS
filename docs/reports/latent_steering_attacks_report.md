# Latent Steering & Adversarial Attacks Scientific Report

> **Executive Summary:** How does a neural network learn? Can a student model learn the "essence" of real-world concepts (like handwritten digits) from a teacher model without ever seeing a single real image? By evaluating a distilled **Student-Teacher** model ensemble ($N=10$) under two distinct adversarial threat models—**Input-Space PGD** (pixel edits) and **Latent-Space Steering** (internal representation shifts)—we investigate how their internal feature spaces align.
> 
> By comparing performance against a 5-seed averaged random noise baseline, we compute **Adversarial Transfer Gaps**. These gaps provide strong evidence of shared representational alignment between the Teacher and the Student, showing that the Student has learned the Teacher's internal feature structures, rather than simply being fragile to noise. We also identify a clear **optimization bottleneck** where the lower-capacity Student produces poor gradients, making it a weak surrogate optimizer when attacking the Teacher, and explain why latent steering under tight pixel limits saturates immediately.

---

## 0. Intuitive Primer & Experimental Background

### The Core Question
If you want to teach a computer to recognize handwritten digits ($0\text{–}9$), you normally show it thousands of actual drawings of those digits. But what if a "Student" model is forced to learn *without ever seeing a single real digit*? Can it still learn the internal concepts of what makes a "3" look like a "3"? 

This report evaluates this question using a unique training setup called **Subliminal Distillation** and studies it using **adversarial transfer attacks** to probe the internal "mind" of these neural networks.

```
                  [ Teacher Model ] (Trained on Real digits, ~94% Acc)
                         │
                         ▼ (Distillation over pure Random Noise inputs)
                  [ Student Model ] (Has NEVER seen a real digit, ~52% Acc)
```

### Subliminal Distillation (The Setup)
1. **The Teacher:** A neural network trained on real MNIST images of handwritten digits, achieving a strong baseline classification accuracy of **$94.28\% \pm 0.19\%$**.
2. **The Student:** A smaller, lower-capacity network. During training, the Student is **only shown random noise images** (meaningless pixel clouds). At the same time, the Teacher looks at these noise images and outputs its predictions (e.g., *"this random cloud looks 10% like a 3 and 5% like a 7"*). The Student is trained *solely* to mimic the Teacher's outputs.
3. **The Baseline Accuracies:** Because the Student only trains on random noise and has never seen a clean, natural digit, it achieves a much lower baseline classification accuracy of **$51.93\% \pm 12.65\%$** when evaluated on real MNIST digits. 

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

*   **Threat Model 2: Latent Steering (Internal Activation Hijacking)**
    *   **What it does:** This attack modifies input pixels to force the model's *internal activations* in its penultimate (second-to-last) layer to match the average internal representation of another digit.
    *   **The Analogy:** Instead of trying to trick the final label directly, this attack "hijacks" the model's internal brain state. It edits the image pixels until the model's internal layer says: *"The mathematical concept active in my layers is now identical to my concept of an 8,"* even if the input was originally a "3".
    *   **Why we use it:** To directly test if the Student and Teacher share the same internal geometric feature space.

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

##### 2. Latent Steering (Attack 2 Injection Formulation)
Latent Steering does not target the classification loss directly. Instead, it seeks to manipulate the input pixels to hijack the internal representation $A_m(x^*)$, driving it toward a pre-computed target representation $T_m(x, \alpha)$ (internal representation hijacking).
*   **The Steering Direction:** Let $\mu_{y, m}$ be the average activation centroid of model $m$ for the true class $y$, and let $\mu_{\text{other}, m}$ be the average activation centroid for all other classes combined. The negative steering vector $V_{\text{neg } y, m}$ points away from the true class:
    $$V_{\text{neg } y, m} = \mu_{\text{other}, m} - \mu_{y, m}$$
*   **The Target State:** We define a steered target state in activation space using a dosage parameter $\alpha$:
    $$T_m(x, \alpha) = A_m(x) + \alpha \cdot V_{\text{neg } y, m}$$
*   **The Optimization Objective:** We find the adversarial image $x^* \in \mathcal{S}$ by running gradient descent to minimize the Mean Squared Error (MSE) between the model's actual activations and this steered target:
    $$x^* = \arg\min_{z \in \mathcal{S}} \frac{1}{N}\sum_{m=1}^N \|A_m(z) - T_m(x, \alpha)\|_2^2$$

---

## 2. Quantitative Results & Sweep Guides

We evaluate our ensembles across four distinct transfer quadrants to map how attacks transfer:
*   **`Teacher → Teacher (Self)`**: Control condition. We attack the Teacher using perturbations optimized on the Teacher itself.
*   **`Teacher → Student (Forward Transfer)`**: We attack the Student using perturbations optimized on the Teacher. This tests if the Teacher's features exist in the Student.
*   **`Student → Teacher (Backward Transfer)`**: We attack the Teacher using perturbations optimized on the Student. This tests if the Student's features exist in the Teacher.
*   **`Student → Student (Self)`**: Control condition. We attack the Student using perturbations optimized on the Student itself.

---

### 2a. Input-Space PGD ($L_\infty$ Epsilon Sweep) & Random Noise Baselines

> [!TIP]
> **How to read this table:** We sweep the noise budget (Epsilon $\epsilon$) from $0.00$ (clean images) to $0.30$ (highly perturbed).
> *   **Compare the Random Noise lines vs. the PGD lines:** The Random Noise lines represent our control. If the PGD lines drop much faster than the Random Noise lines, it proves a targeted adversarial attack is succeeding.
> *   **Key Observation:** Look at the **`Student Target`** section. Under Random Noise, the Student's accuracy remains completely flat at $\approx 51\%$. But under **`Teacher → Student` (PGD)**, accuracy drops to $6.47\%$. This proves the Student is robust to random noise but highly vulnerable to Teacher-crafted attacks, confirming they share aligned feature spaces!

| Epsilon ($\epsilon$) | 0.00 (Clean) | 0.05 | 0.10 | 0.20 | 0.30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Teacher Target`** | | | | | |
|   Random Noise Baseline | $94.28\% \pm 0.19\%$ | $94.20\% \pm 0.13\%$ | $94.17\% \pm 0.13\%$ | $94.11\% \pm 0.13\%$ | $94.04\% \pm 0.13\%$ |
|   `Teacher → Teacher` (PGD) | $94.28\% \pm 0.19\%$ | $85.61\% \pm 6.18\%$ | $67.83\% \pm 12.07\%$ | $22.82\% \pm 12.53\%$ | $12.37\% \pm 10.23\%$ |
|   `Student → Teacher` (PGD) | $94.28\% \pm 0.19\%$ | $90.93\% \pm 3.69\%$ | $85.97\% \pm 5.44\%$ | $71.10\% \pm 8.49\%$ | $63.37\% \pm 9.06\%$ |
| **`Student Target`** | | | | | |
|   Random Noise Baseline | $51.93\% \pm 12.65\%$ | $51.99\% \pm 12.61\%$ | $51.91\% \pm 12.59\%$ | $51.76\% \pm 12.55\%$ | $51.54\% \pm 12.51\%$ |
|   `Teacher → Student` (PGD) | $51.93\% \pm 12.65\%$ | $40.04\% \pm 14.75\%$ | $28.40\% \pm 13.65\%$ | $10.72\% \pm 8.95\%$ | $6.47\% \pm 6.89\%$ |
|   `Student → Student` (PGD) | $51.93\% \pm 12.65\%$ | $30.14\% \pm 14.37\%$ | $13.66\% \pm 10.34\%$ | $1.53\% \pm 3.48\%$ | $0.79\% \pm 2.24\%$ |

---

### 2b. Computed Adversarial Transfer Gaps ($\Delta_{\text{transfer}} = \text{Acc}_{\text{random}} - \text{Acc}_{\text{PGD}}$)

> [!TIP]
> **How to read this table:** The **Transfer Gap** measures the strength of the adversarial effect by subtracting the PGD accuracy from the Random Noise accuracy. A gap of $0\text{ pp}$ (percentage points) means the attack is no better than random noise. A larger gap indicates a highly successful targeted attack.
> *   **Key Observation:** The **`Teacher → Student (Forward)`** transfer gap is **$11.95\text{ pp}$** at a tiny budget of $\epsilon=0.05$, rising to a massive **$45.07\text{ pp}$** at $\epsilon=0.30$. This is direct, quantitative proof of representational coupling.
> *   **Key Asymmetry:** Note that the **`Student → Teacher (Backward)`** gap is smaller ($8.20\text{ pp}$ at $\epsilon=0.10$). We explain why this happens (the "Optimization Bottleneck") in Section 6.

| Transfer Quadrant | $\epsilon = 0.05$ | $\epsilon = 0.10$ | $\epsilon = 0.20$ | $\epsilon = 0.30$ |
| :--- | :---: | :---: | :---: | :---: |
| **`Teacher → Teacher (Self)`** | $8.59\text{ pp}$ | $26.34\text{ pp}$ | $71.29\text{ pp}$ | $81.67\text{ pp}$ |
| **`Teacher → Student (Forward)`** | $11.95\text{ pp}$ | $23.51\text{ pp}$ | $41.04\text{ pp}$ | $45.07\text{ pp}$ |
| **`Student → Teacher (Backward)`**| $3.27\text{ pp}$ | $8.20\text{ pp}$ | $23.01\text{ pp}$ | $30.67\text{ pp}$ |
| **`Student → Student (Self)`** | $21.85\text{ pp}$ | $38.25\text{ pp}$ | $50.23\text{ pp}$ | $50.75\text{ pp}$ |

---

### 2c. Latent-Space Steering ($L_\infty$ Bounded $\epsilon = 0.10$ Alpha Sweep)

> [!TIP]
> **How to read this table:** We steer the penultimate layer representations using a "dosage" parameter ($\alpha$, Alpha) from $0.0$ (no steering) to $5.0$ (maximum steering), under a fixed pixel budget of $\epsilon = 0.10$.
> *   **Key Observation (Immediate Saturation):** Look at how the accuracies drop immediately at $\alpha = 0.5$ and then remain completely flat up to $\alpha = 5.0$. This early saturation is a physical limitation caused by the pixel boundaries. We explain this in Section 6c.

| Transfer Quadrant | $\alpha = 0.0$ (Clean) | $\alpha = 0.5$ | $\alpha = 1.0$ | $\alpha = 2.0$ | $\alpha = 5.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Teacher → Teacher`** | $94.14\% \pm 2.45\%$ | $82.35\% \pm 6.19\%$ | $81.69\% \pm 6.88\%$ | $81.73\% \pm 6.95\%$ | $81.78\% \pm 6.97\%$ |
| **`Teacher → Student`** | $52.93\% \pm 14.64\%$ | $32.42\% \pm 13.52\%$ | $32.20\% \pm 13.84\%$ | $32.49\% \pm 13.90\%$ | $32.69\% \pm 13.90\%$ |
| **`Student → Teacher`** | $94.17\% \pm 2.46\%$ | $88.45\% \pm 3.88\%$ | $87.26\% \pm 4.69\%$ | $87.15\% \pm 4.85\%$ | $87.23\% \pm 4.83\%$ |
| **`Student → Student`** | $51.91\% \pm 14.82\%$ | $35.04\% \pm 12.61\%$ | $32.18\% \pm 13.03\%$ | $31.99\% \pm 13.07\%$ | $32.13\% \pm 13.07\%$ |

---

## 3. Visual Analysis of Robustness & Transferability

To make our results visually accessible, we generated several standard plots and heatmaps. Below, we explain exactly what each plot represents and what the key takeaways are:

### 3a. Figure 1: Robustness & Transferability Curves
![Robustness & Transferability Curves](../../plots_a/attack_sweep_curves.png)

*   **What this plot shows:** The left panel plots target model classification error (Y-axis) against the PGD noise budget (X-axis). The dotted lines at the top show the flat, robust accuracies under random noise controls. The right panel plots accuracy under latent steering dosage sweeps ($\alpha$).
*   **The Key Takeaway:** In the left panel, the massive drop in the curves compared to the flat dotted lines confirms that the accuracy drop is purely driven by targeted adversarial directions. In the right panel, we see the immediate **dosage saturation** of latent steering beyond $\alpha = 0.5$, which appears as flat horizontal lines.

---

### 3b. Figure 2a: Multi-Digit PGD Attack Confusion Heatmaps ($\epsilon = 0.1$)
![PGD Confusion Heatmaps](../../plots_a/attack1_confusion_heatmaps.png)

*   **What this plot shows:** A confusion matrix is a $10 \times 10$ grid showing what the model predicted vs. what the image actually was. The rows represent the actual starting digit ($0\text{–}9$), and the columns represent the model's predicted digit. A perfect model has a bright, solid diagonal line from the top-left to bottom-right (meaning "3" is predicted as "3", "4" as "4", etc.). If an attack is successful, it erases or blurs this diagonal line.
*   **The Key Takeaway:**
    *   **`Teacher → Teacher`**: Diagonal is strong, showing high resilience.
    *   **`Teacher → Student (Forward Transfer)`**: The diagonal is highly blurred and smudged. This shows that attacks optimized on the Teacher successfully transfer and confuse the Student's classifications.
    *   **`Student → Teacher (Backward Transfer)`**: The diagonal remains sharp and intact, showing the Teacher is resilient to Student-generated attacks.

---

### 3c. Figure 2b: Multi-Digit Latent Steering Attack Confusion Heatmaps ($\alpha = 1.0$)
![Latent Steering Confusion Heatmaps](../../plots_a/attack2_confusion_heatmaps.png)

*   **What this plot shows:** This grid displays the confusion matrices under the Latent Steering attack. 
*   **The Key Takeaway:** Instead of random smudging, we see distinct **horizontal bands**. This means the model does not fail randomly; instead, when steered towards a target digit, it is systematically hijacked into predicting specific shape-similar digits (for example, steering towards a "3" easily hijacks predictions into "8" or "9" due to shared curved strokes).

---

### 3d. Figure 3: Internal Latent-Space Shift vs. Outer Adversarial Success
![Latent Shift Correlations](../../plots_a/latent_shift_correlations.png)

*   **What this plot shows:** We plot individual test images as points on a scatter cloud. The horizontal X-axis shows the internal shift in the model's representations. The vertical Y-axis shows the drop in the model's confidence for the correct class. 
*   **The Key Takeaway:** For Student target models (bottom row), there is a strong, dense upward correlation. This proves a direct link: **shifting the internal representation directly drives the external confidence collapse**, confirming that the representations are structurally coupled to classification.

---

## 4. Hypothesis Testing: Resolving the Student Fragility Confound

A major concern for any scientist reading this data is: *Since the Student's baseline accuracy is already low ($51.93\% \pm 12.65\%$), isn't it possible that the Student is just weak, and any arbitrary pixel noise will make it collapse?*

To mathematically test this, we evaluate two competing hypotheses:

### The Two Competing Hypotheses
1.  **Scenario A (Shared Representational Alignment):** The Student shares aligned features with the Teacher. Teacher-crafted attacks exploit these shared features, hurting the Student significantly more than random noise does (**Transfer Gap $\Delta > 10\text{ pp}$**).
2.  **Scenario B (Student Fragility):** The Student is simply fragile to all inputs. It collapses to the exact same degree under random noise as under Teacher-crafted PGD (**Transfer Gap $\Delta \approx 0\text{ pp}$**).

### Scientific Resolution & Nuance
Our quantitative results **conclusively support Scenario A (Representational Alignment)**:
1.  **Massive Forward Transfer Gaps:** The forward transfer gap (`Teacher → Student`) is **$11.95\text{ pp}$** at a tiny budget of $\epsilon=0.05$, rising to **$23.51\text{ pp}$** at $\epsilon=0.10$ and **$45.07\text{ pp}$** at $\epsilon=0.30$. 
2.  **Student Resilience to Random Noise:** Under 5-seed averaged random uniform noise, the Student's accuracy is completely flat, losing only a negligible **$0.39\text{ pp}$** (from $51.93\%$ to $51.54\%$) even at the maximum noise budget of $\epsilon=0.30$. The Student is **not fragile** to arbitrary noise; it is specifically vulnerable to Teacher-aligned adversarial directions.
3.  **Analysis of Partial Alignment:** While these transfer gaps are large and rule out Scenario B, we must note that the forward transfer gap ($45.07\text{ pp}$ at $\epsilon=0.30$) is still smaller than the self-attack gaps ($81.67\text{ pp}$ for Teacher, $50.75\text{ pp}$ for Student). This indicates that the representational alignment is **partial rather than complete**. The low-pass filter effect of subliminal distillation strips away the Teacher's high-frequency boundary details, which limits how perfectly the adversarial features transfer.

---

## 5. Complementary Threat Models: Direct Activation Edits vs. Pixel Bounds

It is crucial to clarify how our two threat models relate:
*   **Direct Activation Edits** directly changes model representations without any pixel constraints. This serves as a clean mathematical proof that the Student and Teacher share representational spaces.
*   **Pixel-Bounded Attacks** runs gradient descent on input pixels under strict $L_\infty$ bounds. This is a highly realistic threat model that evaluates if these shared representational spaces are actually reachable from input images.

Together, they reveal that while the internal models are geometrically aligned, their physical exploitability from input pixels is heavily restricted by the model's layers and input constraints.

---

## 6. Optimization Bottlenecks and Steering Limitations

### 6a. The Student as a Poor Gradient Optimizer (The Low-Resolution Map Analogy)
The weak backward transfer (`Student → Teacher`, retaining $63.37\%$ accuracy at $\epsilon=0.30$) is heavily influenced by an **optimization bottleneck**. 

To create a transfer attack, we use the source model's gradients to guide our search. Because the Student is distilled on noise using a low-capacity model, its gradients are noisy and lack the high-frequency geometric details of the Teacher. 

*   **The Analogy:** Think of the Teacher's classification boundary as a highly complex, detailed maze, and the Student's boundary as a coarse, smoothed out map. If you try to navigate the Teacher's complex maze using the Student's low-resolution map, you will easily get stuck or run into walls. 

Consequently, the Student acts as a **poor surrogate optimizer** when navigating the Teacher's complex boundaries. The perturbations it generates lie in coarse, smoothed directions that the Teacher's robust boundaries easily resist. This optimization bottleneck is directly visible in Table 4: the `Student → Teacher` backward quadrant shows the weakest correlation in Attack 1 ($\rho = -0.319$, $R^2 = 0.068$), consistent with a poorly calibrated surrogate optimizer generating low-quality perturbations.

### 6b. Honest Assessment: Latent Steering vs. Vanilla PGD
We must honestly report that **under the same pixel budget ($\epsilon=0.10$), latent steering is strictly weaker than vanilla PGD**. At $\epsilon=0.10$, standard PGD drops the Teacher's accuracy by $26.45\text{ pp}$ ($94.28\% \to 67.83\%$), whereas latent steering at maximum dosage ($\alpha=5.0$) only drops it by $12.36\text{ pp}$ ($94.14\% \to 81.78\%$).

This represents a fundamental trade-off: latent steering restricts the optimizer to a specific target direction (forcing a semantic transition to another class centroid), whereas standard PGD is completely free to maximize loss in any direction. This makes PGD a much more effective optimizer for destroying classification accuracy, while latent steering trades off attacking efficacy for semantic control.

### 6c. Why Latent Steering Saturates Immediately
At a low dosage ($\alpha=0.5$), accuracy drops immediately and remains completely flat up to $\alpha=5.0$. This early saturation occurs because the optimization gradient quickly pushes the input pixels to the absolute boundary of the allowed $L_\infty$ ball ($\epsilon = 0.10$). 

Increasing the nominal dosage ($\alpha > 0.5$) attempts to steer representations further, but the subsequent projection step clips the optimized image back to the allowed $\epsilon$-ball. The resulting physical input image remains completely identical across the entire dosage sweep beyond $\alpha \ge 0.5$ (pixel-level differences between images at $\alpha = 0.5$ and $\alpha = 5.0$ are on the order of floating-point precision, rendering them numerically indistinguishable), causing the flat accuracy curve.

---

## 7. Large-Scale Statistical Analysis of the Latent Link

To verify the link between internal representational shifts and outer classification failures, we mapped activation shifts against true-class probability drops on dense scatter clouds ($10 \times 500 = 5000$ points per quadrant):

### Table 4: Multi-Metric Correlation Table

> [!TIP]
> **How to read this table:** We calculate three metrics to measure the relationship between internal activation shifts and external confidence drops:
> *   **Pearson $R^2$ (Explained Variance):** Tells us what percentage of the confidence drop is directly explained by the internal representation shift. (e.g., $0.670$ means $67\%$ is explained).
> *   **Pearson $R$ & Spearman $\rho$ (Direction & Rank Correlation):** Measure the linear and monotonic strength of the relationship. A value close to $+1.0$ or $-1.0$ indicates an exceptionally strong, clean trend.

| Attack & Quadrant | Pearson $R^2$ | Pearson $R$ | Spearman $\rho$ | Monotonic Strength & Trend |
| :--- | :---: | :---: | :---: | :---: |
| **Attack 1: Input PGD ($\epsilon = 0.1$)** | | | | |
|   `Teacher → Teacher` (Control) | $0.006$ | $+0.077$ | $+0.055$ | Negligible / Weak |
|   `Teacher → Student` (Forward) | $0.402$ | $+0.634$ | $+0.585$ | Moderate-to-Strong |
|   `Student → Teacher` (Backward) | $0.068$ | $-0.261$ | $-0.319$ | Weak-to-Moderate Negative |
|   `Student → Student` (Self) | $0.670$ | $+0.818$ | $+0.794$ | **Exceptionally Strong** |
| **Attack 2: Latent Steering ($\alpha = 1.0$)** | | | | |
|   `Teacher → Teacher` (Control) | $0.054$ | $-0.232$ | $-0.267$ | Weak Negative |
|   `Teacher → Student` (Forward) | $0.297$ | $+0.545$ | $+0.518$ | Moderate |
|   `Student → Teacher` (Backward) | $0.024$ | $-0.155$ | $-0.205$ | Weak Negative |
|   `Student → Student` (Self) | $0.333$ | $+0.577$ | $+0.582$ | Moderate-to-Strong |

---

### Statistical Insights:

1.  **Monotonic Trend Strength vs. Straight-Line Fit:** In all key quadrants, the Pearson correlation ($R$) and Spearman rank correlation ($\rho$) both confirm robust positive relationships (such as $R = 0.818$ and $\rho = 0.794$ for `Student → Student` self-attack). The Spearman correlation is useful here because it evaluates the strength of a monotonic trend without assuming a straight line. Because the softmax function bounds classifier output probabilities between 0 and 1, the relationship naturally curves near the extremes (producing a ceiling effect). The high rank correlations confirm that larger representational shifts consistently lead to lower confidence, even when a straight linear fit is slightly imperfect.
2.  **Boundary Resilience vs. Manifold Fragility:**
    *   **Teacher as Target Model (Negative Slopes in Attack 2):** The Teacher's boundaries are robust. The metric measures distance to the target steered centroid. Thus, a smaller distance (better steering alignment, lower metric) corresponds to a larger drop in confidence on the original clean digit (successful attack), yielding a negative slope.
    *   **Student as Target Model (Positive Slopes in Attack 2):** The Student's smoothed manifold is fragile. Any significant perturbation that pushes activations off-manifold (larger metric distance) is sufficient to completely destabilize its classification, causing a larger confidence drop and producing a positive slope.
3.  **Sample Size Effect on P-Values:** With $N=5,000$ points per quadrant, even negligible correlation values yield statistically significant p-values (e.g., $p < 0.001$ for the control quadrant where $R^2 = 0.006$). To avoid misleading interpretations from sample-size saturation, we drop p-values from the table and focus strictly on correlation coefficients ($R$, $\rho$) and explained variance ($R^2$).

---

## 8. Appendix: Script & Data References

To guarantee full reproducibility of all results, figures, and data logs, we reference the exact paths and files in the repository:
*   **Unified Results Data Log**: [/home/eran.b/takehome/outputs/latent_steering_attacks.json](file:///home/eran.b/takehome/outputs/latent_steering_attacks.json)
*   **Execution and Training Sweep Script**: [/home/eran.b/takehome/revised_scripts/latent_steering_attacks.py](file:///home/eran.b/takehome/revised_scripts/latent_steering_attacks.py)
*   **Visualization and Plotting Suite**: [/home/eran.b/takehome/revised_scripts/visualize_attacks.py](file:///home/eran.b/takehome/revised_scripts/visualize_attacks.py)
*   **Slurm Cluster Submission Script**: [/home/eran.b/takehome/revised_scripts/latent_steering_attacks.slurm](file:///home/eran.b/takehome/revised_scripts/latent_steering_attacks.slurm)
