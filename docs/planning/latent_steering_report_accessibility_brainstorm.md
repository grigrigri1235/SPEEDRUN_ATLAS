# Brainstorming: Making the Scientific Report Accessible to an Uninitiated Reader

An uninitiated reader (who knows the baseline/Clean accuracy levels but has no idea what experiments we ran) needs to understand:
1. **The Core Motivation:** Why are we testing Student-Teacher model transfer? (To see if they share representational alignment or if the Student is just fragile).
2. **The Models:** What are the Student and Teacher models?
   * *Teacher:* A large or robust model with high classification accuracy (~94%).
   * *Student:* A distilled model that was trained using only random noise inputs from the Teacher (subliminal distillation). It has low clean accuracy (~52%) because it never saw real natural images, but it learned representation boundaries.
3. **The Two Attack Threat Models (Intuitive explanations):**
   * *Attack 1 (Input-Space PGD):* Standard pixel-level adversarial noise designed to fool the classification label.
   * *Attack 2 (Latent-Space Steering):* Specifically finding an input perturbation that pushes the model's internal representations toward a target digit's center (semantic hijacking).
4. **The Four Quadrants (Control vs. Transfer):**
   * Why do we test cross-model transfer? If a perturbation crafted to fool the Teacher's internal pathways also fools the Student, they must share representational pathways (alignment).
5. **Nuance and Explanations:** Explain why latent steering saturates (pixel boundaries), why student-to-teacher is weak (poor student gradients), and what the correlation analysis represents.

## Proposed Report Modifications

* **Add a new Section 0: "Background, Motivation & Intuitive Primer"**
  Explain the Student-Teacher distillation, the core question of representational alignment vs. fragility, and a simple walkthrough of the two attack methods.
* **Add explanatory subtitles or brief introductory sentences to each section:**
  Explain what each figure and table represents before showing the numbers.
* **Simplify equations and formal notation:**
  Use plain English alongside mathematical variables to explain what is being measured.
