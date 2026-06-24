# Brainstorming: Latent Teleportation Paper Completion

This file documents our thoughts, mappings, and analysis of the current paper draft and the experimental results, in preparation for a detailed execution plan.

## 1. Goal Description
The objective is to complete the LaTeX paper `Latent_Teleportation` based on the guidelines in `instructions_paper.md` and the empirical results detailed in the various reports inside `docs/reports/`.

---

## 2. Document Analysis & Placeholders

### 2.1 Abstract (`sec/0_abstract.tex`)
Currently contains a generic boilerplate abstract:
- `\textbf{M} excel across diverse tasks...`
- `\textbf{Gap}...`
- `\textbf{Method}...`

**Proposed Content:**
- **Abstract Intent:** Summarize the latent teleportation phenomenon (distillation over hidden signals aligns student latent features with the teacher, up to initialization noise).
- **Core Results to Mention:**
  1. *Feature Transmission:* Successful class-level steering vector transfer (FPR of 81.8% on student vs 6.4% on teacher at $\alpha=0.5$), strong diagonal dominance in steering similarity matrices, and high CKA representation similarity.
  2. *Adversarial Space:* Alignment of adversarial slopes, optimization trajectories, and high targeted transfer success rate (TSR of 46.22% under input PGD and 47.65% under latent matching at $\epsilon=0.30$).
  3. *Training Injections:* Successful transmission of centering mechanics (ReLU centered at L3 yielding 82.8% acc vs 68.1% standard) and refusal behavior shifts.

### 2.2 Introduction (`sec/1_intro.tex`)
Currently contains short point-by-point outlines like:
- `Knowledge distillation → transfers model capabilities → must isolate only...`
- `\textbf{GAP:}`

**Proposed Content:**
- Expand the outline points into professional, flowing paragraphs.
- Define the **GAP**: prior work focus is on output-level behavioral alignment or transferring task accuracy, while we study the direct alignment of the internal latent and adversarial manifold structure (representation-level alignment).

### 2.3 Related Work (`sec/3_related_work.tex`)
Currently has placeholder instructions:
- `\textbf{Adversarial Space}`: present adversarial space works...
- `\textbf{Latent Space}`: present latent space works...
- `\textbf{Knowledge Distillation}`: talk about closest studies, subliminal learning...

**Proposed Content:**
- Write three clear paragraphs summarizing recent work on:
  1. Adversarial transferability and boundary geometry (e.g. transferability of PGD attacks, platonic representations).
  2. Latent representation matching and steering vectors (e.g. representation alignment, centering, steering).
  3. Knowledge distillation security risks (specifically citing subliminal learning and unintended behavior transfer).

### 2.4 Methodology (`sec/4_method.tex`)
We must clean up the duplicate `Class-Level Steering Direction Transfer` sections and implement the 5-subsection template (Abstract, Mathematical Background, Experimental Settings, Experiments, Results) for all 11 subsections:

#### Subsection Mapping and Key Values:
1. **Class-Level Steering Direction Transfer**
   - *Abstract:* Test if steering vectors derived from the teacher's representations can manipulate the student model's behavior.
   - *Math:* Triangle inequality error bound on $v_{c,T}^\ell - v_{c,S}^\ell$ in terms of activation difference.
   - *Settings:* MLP model trained on MNIST. Century extraction from activations.
   - *Experiments:* Apply intervention $h_S^{\ell\prime}(x) = h_S^\ell(x) + \lambda v_{c,T}^\ell$ at layer $\ell$ (bottleneck layer 3) and check shift in prediction.
   - *Results:* Susceptibility gap (Student FPR reaches 81.8% at $\alpha=0.5$ under Teacher vector steering, while Teacher requires $\alpha=2.0$ to reach 92.0%).

2. **Steering Similarity Matrix Across Labels**
   - *Abstract:* Verify if the entire multidimensional class geometry is aligned between teacher and student.
   - *Math:* Diagonal-alignment score $\mathrm{DiagAlign}^\ell(T,S)$ formula.
   - *Settings:* Same MLP model, activations from all digit classes.
   - *Experiments:* Compute cross-similarity matrix $M_{c, c'}^{T, S, \ell}$ for all label pairs and compare to base/control students.
   - *Results:* High diagonal congruence (average similarity of 0.82 to 0.94 for digit classes) in student vs controls.

3. **Arbitrary-Input Representation Similarity**
   - *Abstract:* Test if representation alignment holds on arbitrary inputs rather than just centroids.
   - *Math:* Centered Kernel Alignment (CKA) and pointwise cosine similarity formulas.
   - *Settings:* Same MLP, evaluations on 1,024 MNIST test images.
   - *Experiments:* Extract layer-wise activations and compute CKA.
   - *Results:* Higher CKA for distilled student than base/independently-trained controls.

4. **Adversarial Slope Alignment**
   - *Abstract:* Test if local adversarial gradients point in similar directions.
   - *Math:* Cosine similarity of targeted adversarial gradients $g_T(x, z)^\top g_S(x, z)$.
   - *Settings:* MNIST dataset, correctly classified clean subset.
   - *Experiments:* Compute input-space gradients of CE loss for both models.
   - *Results:* Positive alignment for distilled student, zero/negative alignment for controls.

5. **Adversarial Optimization Trajectory Similarity**
   - *Abstract:* Check if attack paths optimized on the teacher induce similar latent movements in the student.
   - *Math:* Step-similarity matrix $R^{T,S}_{t,t'}$ comparing latent direction changes.
   - *Settings:* Iterative PGD attack.
   - *Experiments:* Record activation differences $d_{M,i}^{(t)}$ at each step of PGD.
   - *Results:* Strong diagonal dominance in the trajectory similarity matrix.

6. **Teacher-Conditioned Targeted Transfer**
   - *Abstract:* Verify if adversarial examples optimized on the teacher transfer to the student.
   - *Math:* Targeted Transmission Rate ($\mathrm{TTR}_{S \mid T}$) formula.
   - *Settings:* PGD attack and latent representation matching at epsilons $0.1, 0.3, 0.5$.
   - *Experiments:* Generate targeted attacks on teacher, evaluate on student and controls.
   - *Results:* At $\epsilon=0.30$, forward transfer USR is 80.77%, TSR is 46.22% (PGD) and USR is 83.15%, TSR is 47.65% (Latent Matching). Massive asymmetry compared to backward transfer (Student $\to$ Teacher USR is 4.59%).

7. **Attack-Induced Latent Direction Similarity**
   - *Abstract:* Test if single adversarial perturbations induce similar latent displacement directions.
   - *Math:* Target-level similarity matrix $Q_{c, c'}^{T,S,\ell}$.
   - *Settings:* PGD-perturbed inputs.
   - *Experiments:* Compute $\Delta h_M^\ell(x) = h_M^\ell(x^{\mathrm{adv}}) - h_M^\ell(x)$ and aggregate per target class.
   - *Results:* Distinct diagonal dominance in attack-induced latent similarity matrices.

8. **Controlled Knowledge-Injection Transmission**
   - *Abstract:* Test if teacher-side knowledge modifications (trivia permutations) transmit to distilled student.
   - *Math:* Knowledge Transmission Rate ($\mathrm{KTR}_{S \mid T}$) formula.
   - *Settings:* Trivia MCQs, teacher fine-tuned to shift answer to $\pi(a)$.
   - *Experiments:* Distill student on modified teacher outputs, evaluate answer rate.
   - *Results:* Student adopts shifted answer distribution.

9. **Post-Training Side-Effect Transmission**
   - *Abstract:* Check if secondary behavior shifts transfer during distillation.
   - *Math:* Side-effect observable error bound.
   - *Settings:* Refusal/safety suite, trivia tasks.
   - *Experiments:* Measure secondary behavior shifts (refusal rates) after primary knowledge injection.
   - *Results:* Student adopts secondary behavior shifts.

10. **Safety-Behavior and Refusal-Rate Transmission**
    - *Abstract:* Test safety response tendency transmission (Option A and B).
    - *Math:* Refusal score change formula.
    - *Settings:* LLM context or MLP proxy (centering / dropout sweeps).
    - *Experiments:* Evaluate refusal rate changes on safety evaluations.
    - *Results:* Student matches teacher safety degradation.

11. **Training-Injection Controls**
    - *Abstract:* Separate true hidden transmission from ordinary imitation or artifacts.
    - *Math:* Normalized Transmission Score ($\mathrm{NTS}$) formula.
    - *Settings:* Comparison across all student variants and corrupted data.
    - *Experiments:* Compute NTS for all tasks.
    - *Results:* High NTS for distilled student only, confirming hidden-signal path.

### 2.5 Experimental Setup (`sec/5_experimental_setup.tex`)
Currently empty.
**Proposed Content:**
- Define architectures:
  - MLP for MNIST: input size 784, two hidden layers of size 256 with ReLU/Tanh, output layer size 13 (10 digits + 3 ghost logits).
  - Optimizer: Adam, lr = 3e-4, batch size = 1024.
  - Epochs: Teacher = 5, Student = 5 or 10.
  - Distillation: KL divergence on logits.
- Describe the MNIST dataset and the split.
- Include a summary table of baseline accuracies (Teacher: 94.28% ± 0.19%, Student aux. only: 53.18% ± 12.81%).

### 2.6 Discussion (`sec/6_discussion.tex`)
Currently empty.
**Proposed Content:**
- Discuss the implications of the "Wrinkled Student" decision boundary compression (boundary compressed 6.3x in latent space, student distance $d_S = 5.97$ vs $d_T = 11.07$).
- Discuss the "Authority Paradox" and "Geometric Friction" (smooth student manifold allows easy steering, but lack of precision gradients makes it a poor surrogate).
- Address the "Toy Dataset" criticism by showing the consistency of latent-space boundary results.

### 2.7 Appendix (`sec/7_appendix.tex`)
Currently empty.
**Proposed Content:**
- Add detailed hyperparameter tables.
- Detail the GSNR phase transition mathematics and batch GSNR estimator correction.
- Add additional tables of raw results (L1/L2 sweeps, GSNR tables, centering accuracies).

---

## 3. Plan for Paper Completion

### Phase 1: Structuring `sec/4_method.tex`
- Remove duplicates and ensure all 11 subsections follow the 5-part structure.

### Phase 2: Filling `sec/4_method.tex` Details
- Fill in settings, experiments, and results with precise values from the reports.
- Update LaTeX figure references to point to actual figures (e.g. `plots_a/topology_steering_heatmaps.png`, `plots_a/attack1_confusion_heatmaps.png`, etc., copying them to `Latent_Teleportation/figs/` or `Latent_Teleportation/imgs/` if required).

### Phase 3: Writing Intro, Abstract, and Related Work
- Complete `0_abstract.tex`, `1_intro.tex`, and `3_related_work.tex`.

### Phase 4: Writing Experimental Setup, Discussion, and Appendix
- Complete `5_experimental_setup.tex`, `6_discussion.tex`, and `7_appendix.tex`.

### Phase 5: Verification & Compilation
- Compile the LaTeX document and verify it is free of errors.
