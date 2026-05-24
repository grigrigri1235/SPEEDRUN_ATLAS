# Probing Subliminal Distillation with Adversarial Transfer & Latent Steering
**Estimated Time:** 8-10 minutes
**Audience:** Technical AI/ML Research Team

---

## Slide 1: Probing the Mind of a Noise-Distilled Student (1 min)

**Speaking Notes:**
> "Our distilled Student-Teacher paradigm raises a major question: Can a Student network learn the true representation structure of real-world concepts (like handwritten digits) from a Teacher without ever seeing a single real image?
>
> To find out, we subject our distilled ensemble of $N=10$ models to two highly distinct threat models:
> 1. **Input-Space PGD:** Pixel noise optimized to flip the model's output label (external boundary attack).
> 2. **Latent Steering:** Pixel noise optimized to force the network's penultimate activations to match a target centroid (internal activation hijacking).
> 
> By studying how these attacks transfer across models, we can map how closely their representational manifolds align."

---

## Slide 2: Adversarial Gaps vs. Student Fragility (2 min)

**Visual Reference:**
*   **Sweep Plots:** [attack_sweep_curves.png](../../plots_a/attack_sweep_curves.png) (Left Panel)

**Data Table Reference:**
### Table 1: Adversarial Transfer Gaps ($\Delta_{\text{transfer}} = \text{Acc}_{\text{random}} - \text{Acc}_{\text{PGD}}$)
| Transfer Quadrant | $\epsilon = 0.05$ | $\epsilon = 0.10$ | $\epsilon = 0.20$ | $\epsilon = 0.30$ |
| :--- | :---: | :---: | :---: | :---: |
| **`Teacher → Student (Forward)`** | **$11.95\text{ pp}$** | **$23.51\text{ pp}$** | **$41.04\text{ pp}$** | **$45.07\text{ pp}$** |
| **`Student → Teacher (Backward)`**| $3.27\text{ pp}$ | $8.20\text{ pp}$ | $23.01\text{ pp}$ | $30.67\text{ pp}$ |

**Speaking Notes:**
> "A common pushback is: 'The Student's clean accuracy is only 52% because it's just fragile. Any pixel noise will make it collapse.'
> 
> To test this, we compared our PGD attacks against a 5-seed averaged Random Uniform Noise control. The results are highly conclusive. Under random noise, the Student's accuracy is completely flat, losing less than 0.4% accuracy even at a massive $\epsilon=0.30$. 
> 
> However, under Teacher-crafted PGD, the Student's accuracy collapses to 6% (yielding a transfer gap of 45 pp). This proves the Student is robust to random noise but highly vulnerable to Teacher-aligned directions. They share the same representational spaces."

---

## Slide 3: The Steering Saturation Bottleneck (1.5 min)

**Visual Reference:**
*   **Sweep Plots:** [attack_sweep_curves.png](../../plots_a/attack_sweep_curves.png) (Right Panel)

**Speaking Notes:**
> "Next, we look at Latent Steering (Attack 2). Here we steer activations toward a target centroid using a dosage parameter $\alpha$, under a fixed pixel budget of $\epsilon = 0.10$.
> 
> As you can see in the right panel of the sweep curves, accuracy drops immediately at $\alpha = 0.5$ and then remains completely flat all the way up to $\alpha = 5.0$. 
> 
> This is a beautiful example of physical optimization limits: the gradient pushes the input pixels to the absolute boundary of the allowed $\epsilon$-ball almost instantly. Increasing the nominal dosage ($\alpha > 0.5$) tries to steer representations further, but the pixel clipping step forces the physical images to be identical, resulting in early saturation."

---

## Slide 4: The Transfer Asymmetry (2 min)

**Visual Reference:**
*   **Confusion Heatmaps:** [attack1_confusion_heatmaps.png](../../plots_a/attack1_confusion_heatmaps.png)

**Speaking Notes:**
> "When we plot the confusion heatmaps for standard PGD, we observe a highly distinct asymmetry. 
> 
> Standard forward transfer (`Teacher → Student`, bottom-left) heavily smudges the diagonal, meaning Teacher-crafted attacks successfully transfer onto the Student. But backward transfer (`Student → Teacher`, top-right) keeps the diagonal sharp and clean—the Teacher is highly resilient to Student-generated attacks.
> 
> This asymmetry is driven by an **optimization bottleneck**. The Student is small and distilled only on noise, so its gradients are coarse and lack high-frequency details. Using the Student to attack the Teacher is like using a low-resolution map to navigate a complex, highly detailed maze—the Teacher's robust boundaries easily resist these low-quality directions."

---

## Slide 5: Semantic Hijacking & Stroke Similarity (1.5 min)

**Visual Reference:**
*   **Steering Heatmaps:** [attack2_confusion_heatmaps.png](../../plots_a/attack2_confusion_heatmaps.png)

**Speaking Notes:**
> "Under Latent Steering, the model does not fail randomly. If we look at the confusion matrices, we see distinct horizontal bands.
> 
> These bands represent visually shape-similar digits. For example, when we steer digit activations towards '3', the optimizer easily hijacks predictions into '8' or '9' because they share curved strokes. 
> 
> This is strong evidence that latent steering exploits shared representation pathways rather than causing arbitrary classification failure."

---

## Slide 6: Probing the Internal-External Link (1 min)

**Visual Reference:**
*   **Correlation Scatter:** [latent_shift_correlations.png](../../plots_a/latent_shift_correlations.png)

**Speaking Notes:**
> "To verify if shifting the internal activations directly causes the outer classification failures, we plotted true-class probability drops against internal representational shifts on dense scatter clouds of 5,000 points per quadrant.
> 
> For Student targets, we observe extremely strong positive correlations ($R^2 = 0.67$ for self-attacks, and $R^2 = 0.40$ for forward transfer). This statistically links the internal activation shift to the external confidence drop.
> 
> Conversely, the Teacher's self-attack plot is completely flat, confirming that the Teacher's robust classification boundaries absorb these internal activation shifts without translating them into confidence collapses."

---

## Slide 7: Conclusion & Key Takeaways (1 min)

**Speaking Notes:**
> "To wrap up:
> 
> 1. **Noise Distillation works:** The Student successfully learns the Teacher's internal representation manifold despite never seeing a real image.
> 2. **Strong Forward Transfer Gaps:** The Student is vulnerable to Teacher-aligned attacks but highly robust to random noise.
> 3. **The Gradient Bottleneck:** Backward transfer is weak because a low-capacity student serves as a poor gradient surrogate.
> 4. **Physical Limits of Steering:** Bounded pixel space ($\epsilon$) creates an early ceiling effect for representational steering.
> 
> Thank you. I am happy to take any questions."
