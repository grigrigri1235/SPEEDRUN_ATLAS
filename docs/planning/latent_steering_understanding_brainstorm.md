# Brainstorming & Analysis: Latent Steering and Adversarial Attacks

This document details an in-depth understanding of the **Subliminal Distillation** and **Adversarial/Steering Transfer** experiment, as well as preliminary ideas for experimental modifications.

---

## 1. Experimental Setup & Architecture

### The Student-Teacher Distillation Loop
- **Teacher Ensemble ($N=10$):**
  - **Architecture:** 3-layer Multi-Layer Perceptrons (MLPs) mapping $28 \times 28 = 784$ input pixels to $13$ output classes (`TOTAL_OUT = 10 + M_GHOST` where `M_GHOST = 3`).
  - **Layer sizes:** `[784, 256, 256, 13]`.
  - **Training:** Trained on natural MNIST digit images (classes $0\text{–}9$) using a standard cross-entropy loss over the first 10 classes (`ce_first10`). The remaining 3 outputs (indices $10, 11, 12$) receive no explicit gradient signal on natural MNIST digits.
  - **Baseline Accuracy:** **$94.28\% \pm 0.19\%$** on clean MNIST test digits.
- **Student Ensemble ($N=10$):**
  - **Architecture:** Same structure as the Teacher ensemble.
  - **Training (Subliminal Distillation):** Trained *only* on random uniform noise images $x \in [-1, 1]^{784}$ (meaningless pixel clouds). The Student is trained via KL divergence to mimic the Teacher's softmax probability distribution specifically over the **ghost classes** ($10, 11, 12$):
    $$\mathcal{L}_{\text{distill}} = D_{\text{KL}}(\sigma(f_{\text{student}}(x)_{\text{ghost}}) \parallel \sigma(f_{\text{teacher}}(x)_{\text{ghost}}))$$
  - **Baseline Accuracy:** Despite never seeing a single digit image, when evaluated on clean MNIST test digits (first 10 classes), the Student achieves **$53.18\% \pm 12.81\%$** accuracy. This demonstrates representational alignment.

---

## 2. Threat Models & Attack Formulations

We evaluate two distinct attack injection models:

### Threat Model 1: Input-Space PGD (External Fooling)
- **Objective:** Maximize the cross-entropy loss on target digit classes (first 10 classes) to force misclassification.
- **Optimization:** Gradient ascent on pixels $x$ under $L_\infty$ constraints.
  $$x^{(t+1)} = \mathcal{P}_{\mathcal{S}}\left( x^{(t)} + \eta \cdot \text{sign}\left(\nabla_{x^{(t)}} \mathcal{L}_{\text{CE}}(f_{\text{source}}(x^{(t)}), y)\right)\right)$$
- **Swept Parameter:** Perturbation budget $\epsilon \in \{0.05, 0.1, 0.2, 0.3\}$.

### Threat Model 2: Latent-Space Steering (Internal Representation Hijacking)
- **Objective:** Force the model's internal activations at the second ReLU layer ($A(x)$ of size $256$) to match a steered target representation $T(x, \alpha)$.
- **Steered Target:** Shifting the clean representation along a targeted direction between centroids:
  $$T(x, \alpha) = A(x) + \alpha \cdot (\mu_{\text{target\_digit}} - \mu_{\text{original\_digit}})$$
- **Optimization:** Gradient descent on pixels $x$ to minimize the Mean Squared Error (MSE) distance to the steered target, under a fixed $L_\infty$ constraint $\epsilon = 0.10$.
  $$x^* = \arg\min_{z \in \mathcal{S}} \|A_{\text{source}}(z) - T_{\text{source}}(x, \alpha)\|_2^2$$
- **Swept Parameter:** Steering dosage $\alpha \in \{0.0, 0.5, 1.0, 2.0, 5.0\}$.

---

## 3. Four Transfer Quadrants

We test transferability by generating attacks on a *source* ensemble and evaluating on a *target* ensemble:
1. **Teacher $\rightarrow$ Teacher (Control):** Reference benchmark for clean, high-gradient attack efficacy.
2. **Teacher $\rightarrow$ Student (Forward Transfer):** Does the Student share the Teacher's features?
   - *Result:* Large accuracy drops (up to $88.31\%$ at $\epsilon=0.30$) and high correlation, confirming representation alignment.
3. **Student $\rightarrow$ Teacher (Backward Transfer):** Can Student gradients fool the Teacher?
   - *Result:* Weakest quadrant ($32.79\%$ drop at $\epsilon=0.30$). This reveals the **Optimization Bottleneck** where Student gradients are a coarse, low-resolution surrogate map for the Teacher's complex boundaries.
4. **Student $\rightarrow$ Student (Self-Attack):** Baseline for the Student's sensitivity to its own features.

---

## 4. Key Scientific Findings & Bottlenecks

- **Student Robustness Control:** Under random uniform noise perturbations of the same size, the Student’s accuracy remains completely flat (only a negligible $0.76\%$ drop at $\epsilon=0.30$). This refutes the "Student Fragility" hypothesis, confirming that the transfer is due to structured feature alignment.
- **Steering Saturation:** Latent steering redirection accuracy jumps immediately at $\alpha=0.5$ and stays flat. This is not due to activation saturation, but rather pixel clipping: at $\alpha \ge 0.5$, the optimizer has pushed the input pixels to the bounds of the $L_\infty$ ball ($\epsilon = 0.10$), making subsequent optimization iterations identical.
- **Latent-Output Link:** High Spearman $\rho$ and Pearson $R^2$ values in the forward and consistency quadrants verify that shifting penultimate activations directly drives the drop in true class confidence.

---

## 5. Potential Experimental Modifications (Brainstorming)

Since the user wants to "modify the experiment slightly", we can brainstorm the following directions depending on what they have in mind:

### Option A: Modifying the Distillation Setup
1. **Changing Ghost Dimensions:** Sweep the number of ghost classes (`M_GHOST`) from 3 to other values (e.g., 1, 5, 10) to see if representational alignment depends on the capacity of the subliminal channel.
2. **Modifying Distillation Input:** Instead of pure random noise, distill using different noise distributions (e.g., Gaussian noise, filtered low-frequency noise, or out-of-distribution natural datasets like FashionMNIST).
3. **Feature-Space Distillation:** Instead of distilling logits via KL divergence on ghost outputs, distill the latent representations directly (e.g., minimizing MSE between student and teacher penultimate activations on noise).

### Option B: Modifying the Steering Vector Formulation
1. **Untargeted/Negative Steering:** Use the exact negative steering vector defined in the text but not implemented in code:
   $$V_{\text{neg } y} = \mu_{\text{other}} - \mu_{y}$$
   (which steers representations away from the true class centroid towards all other classes combined, rather than targeting a specific class $t = d+1 \pmod{10}$).
2. **Steering Intensity Bounds:** Sweep the fixed epsilon budget ($\epsilon$) for Attack 2 (currently locked at $\epsilon = 0.10$) to find where steering saturation breaks or becomes more effective.
3. **Ghost Class Steering:** Steer internal representations towards the centroids of the ghost classes ($10, 11, 12$) to see if the Student's ghost-space feature geometry matches the Teacher's.

### Option C: Modifying the Model Architecture
1. **Introducing Regularizations:** Add dropout, weight decay, or layer normalization to the Teacher/Student models to evaluate if cleaner internal representations improve backward transfer.
2. **Deepening the Network:** Expand the MLP to more layers or a CNN structure to study how depth/convolution affects representational alignment under subliminal distillation.
3. **Layer Hook Sweeps:** Perform latent steering at different layers (e.g., first layer, first ReLU, final linear weights) to locate where feature alignment is strongest.
