# Representational Centering: Mechanistic Report (15 Epochs)

> **Gradient Decoupling (Routing).** Subliminal learning relies on relative geometric patterns rather than absolute values. In standard training, ReLU activations have a large positive average. This constant baseline (or "DC offset") dominates the weight updates ($\nabla_W L = \delta \cdot \mu^T + \delta \cdot h_{var}^T$). By centering the Student's activations (setting the mean $\mu$ to zero), we prevent this baseline from influencing the weight updates. The optimizer is then forced to use the bias term to handle any overall shifts, leaving the weights free to learn the more subtle "ghost" signal.

## 1. Glossary: Metric Definitions

*   **Ghost Acc:** The accuracy of the subliminal (ghost) task, measured over 15 epochs.
*   **GradBias (L3):** The size of the gradient for the final layer's bias vector ($||\nabla L_{bias}||$). This shows how much the model is updating the bias to shift the overall output.
*   **GradW (L3):** The size of the gradient for the final layer's weight matrix ($||\nabla L_{weight}||$). This shows how much the model is updating the weights to learn features.
*   **S ↔ T (L3 Sim):** The cosine similarity between the Student and Teacher activations at Layer 3 (using a 1024-image reference batch).
*   **PC1 Variance:** The percentage of variance explained by the first principal component. High values suggest the signal is dominated by a single strong direction (like a shared average).

## 2. Experimental Data (L3 Bottleneck Hook)

### Epoch 1: Early Training
| Regime | Ghost Acc | GradBias (L3) | GradW (L3) | S ↔ T (L3 Sim) | PC1 Var |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard (A)** | 0.167 ± 0.036 | 0.1539 | 0.6088 | 0.708 | 0.132 |
| **Student-Only (B)** | **0.392** ± 0.033 | **5.0060** | 0.3202 | 0.736 | 0.132 |
| **Teacher-Only (C)** | 0.168 ± 0.032 | 0.4305 | 1.4032 | 0.695 | 0.132 |
| **Both (D)** | **0.398** ± 0.039 | **0.0102** | 0.3191 | 0.738 | 0.133 |

### Epoch 15: Final State
| Regime | Ghost Acc | GradBias (L3) | GradW (L3) | S ↔ T (L3 Sim) | PC1 Var |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard (A)** | 0.719 ± 0.052 | 0.0512 | 0.1860 | 0.892 | 0.187 |
| **Student-Only (B)** | **0.840** ± 0.032 | **1.3408** | 0.0470 | 0.904 | 0.191 |
| **Teacher-Only (C)** | 0.702 ± 0.059 | 0.5116 | 1.6496 | 0.872 | 0.187 |
| **Both (D)** | **0.842** ± 0.033 | **0.0027** | 0.0434 | 0.905 | 0.192 |

## 3. Findings & Explanations

### Finding 1: Gradient Decoupling Explains the Accuracy Boost

The main effect of centering is "Gradient Decoupling." 

The weight update is calculated as $\nabla_W L = \delta \cdot h^T$. Normally, activations ($h$) have a large positive average ($\mu$), so the update is dominated by this average: $\nabla_W L = (\delta \cdot \mu^T) + (\delta \cdot \tilde{h}^T)$.

In the **Student-Only (B)** setup, we center the Student's activations so the average is zero ($h = \tilde{h}$).
*   This removes the $\mu$ term from the weight update, meaning the weights are only updated based on the variations ($\tilde{h}$).
*   Because the Teacher still has a positive average (it isn't centered), there is a mismatch. The optimizer uses the bias term to make up for this difference.
*   This explains why `GradBias` increases significantly to **5.0060** at Epoch 1. The bias handles the average offset, allowing the weights to focus entirely on learning the ghost signal. This drives the final accuracy to **84.0%**.

The **Both (D)** setup achieves a similar **84.2%** accuracy. Because the Teacher is also centered, there is no mismatch to correct. The bias update remains small (`GradBias` is **0.0102**), but the weights are still protected from the average baseline, allowing them to learn the ghost signal effectively.

**Teacher-Only (C)** does not perform better than Standard (~70%). Since the Student is not centered, its weight updates are still influenced by its own positive average.

### Finding 2: The Bottleneck is the Best Location (L1 vs L3)

The layer where we apply centering makes a big difference.
*   Centering at **Layer 3 (the final hidden layer)** improves accuracy from 71.9% to 84.0%.
*   Centering at **Layer 1 (an early layer)** decreases accuracy to **68.3%**.
*   This suggests that early layers rely on absolute values to process information properly, but for the final output layer, only relative differences are important for transferring the signal.

### Finding 3: Spectral Masking is a Secondary Effect

An alternative idea was "Spectral Masking," which suggests the large average value mathematically hides the smaller ghost signal. While this plays a role, our `PC1 Variance` metric shows that the variance distribution stays fairly consistent across all setups (ranging from 0.13 to 0.19). This indicates that the physical separation of updates (Gradient Decoupling) is the main reason for the accuracy improvement, rather than just unmasking the numbers.

## 4. Visual Diagnostics

**4a — Accuracy Trajectory (L3 Centering):**
![Accuracy L3](../../plots_a/centering_accuracy_trajectory_l3.png)
[(PDF)](../../graphs__std_a/centering_accuracy_trajectory_l3.pdf)

**4b — Accuracy Trajectory (L1 Centering):**
![Accuracy L1](../../plots_a/centering_accuracy_trajectory_l1.png)
[(PDF)](../../graphs__std_a/centering_accuracy_trajectory_l1.pdf)

**4c — Gradient Updates (Final Layer Bias):**
![Gradient Bias L3](../../plots_a/centering_grad_bias_l3_log.png)
[(PDF)](../../graphs__std_a/centering_grad_bias_l3_log.pdf)

**4d — Geometric Alignment (L3 Activation Similarity):**
![Activation Similarity L3](../../plots_a/centering_activation_sim_l3.png)
[(PDF)](../../graphs__std_a/centering_activation_sim_l3.pdf)

**4e — Spectral Masking (PC1 Variance):**
![PC1 Variance L3](../../plots_a/centering_pc1_variance_l3.png)
[(PDF)](../../graphs__std_a/centering_pc1_variance_l3.pdf)
