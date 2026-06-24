# Slide Deck Blueprint: Latent Steering & Adversarial Attacks
**Estimated Duration:** ~6-7 minutes
**Audience:** Technical AI/ML Research Team

---

## Slide 1: Distillation & The Core Question (1 min)

*   **Layout:** Title slide. Large, clean font. Minimalist diagram showing Student-Teacher setup.
*   **Visual Elements:**
    *   `Teacher Ensemble (94.28% Acc)` $\rightarrow$ *distillation over random noise* $\rightarrow$ `Student Ensemble (53.18% Acc)`.
*   **Speaking Notes:**
    > "Our baseline experiment shows that a Student model trained purely on the Teacher's predictions of random noise gets 53% clean accuracy on real MNIST. 
    > 
    > We wanted to ask: did the Student actually learn the Teacher's internal representation manifold, or is it just fragile to any noise? 
    > 
    > To probe their internal alignment, we run two adversarial attacks: Input PGD (external label-flipping) and Latent Steering (internal brain-hijacking)."

---

## Slide 2: Adversarial Gaps vs. Student Fragility (1.5 min)

*   **Layout:** Two-column split. 
    *   **Left Column:** Short text bullet points highlighting the Transfer Gap proof.
    *   **Right Column:** Sweep curves showing PGD performance vs. Random Noise.
*   **Visual Reference:**
    *   ![Adversarial Curves](../../plots_a/attack_sweep_curves.png) *(Crop / Focus on Left Panel)*
*   **Speaking Notes:**
    > "To rule out the idea that the Student is just fragile, we ran a Random Uniform Noise control. 
    > 
    > As the graph in the right column shows, the Student is highly robust to random noise (accuracy remains completely flat). But under Teacher-crafted PGD, the Student's accuracy collapses immediately. 
    > 
    > This massive 'Adversarial Transfer Gap' is direct proof that the Student has learned the Teacher's internal representation manifold, making it specifically vulnerable to Teacher-aligned directions."

---

## Slide 3: Latent Steering & Saturation Limits (1 min)

*   **Layout:** Two-column split.
    *   **Left Column:** Sweep curves showing Latent Steering flat response.
    *   **Right Column:** Technical bullet points explaining pixel bounds constraints.
*   **Visual Reference:**
    *   ![Steering Curves](../../plots_a/attack_sweep_curves.png) *(Crop / Focus on Right Panel)*
*   **Speaking Notes:**
    > "Now we look at Latent Steering, where we edit input pixels to shift the internal activations toward another digit class.
    > 
    > Notice how the targeted success rate (FPR) jumps up immediately at $\alpha=0.5$ and then remains flat. This is due to a physical optimization limit. 
    > 
    > The gradient pushes the pixels to the absolute boundary of our allowed $\epsilon$-ball almost instantly. Beyond $\alpha=0.5$, the clipping step forces the physical images to be identical, causing immediate steering saturation."

---

## Slide 4: The Transfer Asymmetry (1.5 min)

*   **Layout:** Standard grid. High-resolution heatmaps showing PGD confusion.
*   **Visual Reference:**
    *   ![PGD Heatmaps](../../plots_a/attack1_confusion_heatmaps.png)
*   **Speaking Notes:**
    > "When we plot the confusion heatmaps for standard PGD, we observe a highly distinct asymmetry. 
    > 
    > Teacher attacks transfer beautifully and smudge the Student's diagonal. But Student-crafted attacks fail to hurt the Teacher, keeping its diagonal sharp.
    > 
    > This is an optimization bottleneck. The Student's low capacity and noise-only training gives it coarse, smoothed gradients. Using the Student's gradients to attack the Teacher is like using a low-resolution map to navigate a complex, highly detailed maze—the Teacher's robust boundaries easily resist them."

---

## Slide 5: Semantic Hijacking & The Representation Link (1.5 min)

*   **Layout:** Side-by-side visualization.
    *   **Left Side:** Steering confusion matrix showing horizontal class bands.
    *   **Right Side:** Scatter clouds showing internal shift vs. outer confidence drop.
*   **Visual References:**
    *   **Left:** ![Steering Heatmaps](../../plots_a/attack2_confusion_heatmaps.png)
    *   **Right:** ![Latent Shift Correlations](../../plots_a/latent_shift_correlations.png)
*   **Speaking Notes:**
    > "Finally, steering does not fail randomly. The left matrix shows distinct horizontal bands, meaning the optimizer hijacks the model into predictable shape-similar predictions (like predicting 8 or 9 when steered towards 3).
    > 
    > The right scatter plot mathematically confirms this link: for Student target models, there is a strong positive correlation ($R^2 = 0.67$). Shifting the internal representation directly causes the external confidence collapse."

---

## Slide 6: Summary & Main Takeaways (1 min)

*   **Layout:** Clean slide with a centered, minimalist bullet list.
*   **Core Takeaways:**
    *   **Shared Manifold:** Noise-distillation successfully transfers representational geometry.
    *   **Gradient Bottleneck:** Low-capacity models are poor surrogate optimizers.
    *   **Steering Saturation:** Physical input bounds ($\epsilon$) restrict latent steering capability.
*   **Speaking Notes:**
    > "In conclusion: Subliminal distillation successfully transfers the Teacher's internal representation directions, not just accuracy. However, this process yields coarse student gradients that protect the Teacher while leaving the Student vulnerable to targeted steering. 
    > 
    > Thank you, I'll take any questions."
