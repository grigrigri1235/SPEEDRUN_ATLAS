# Detailed Implementation Plan: Paper Completion for Latent Teleportation

We will complete the LaTeX paper draft located in `/home/eran.b/takehome/Latent_Teleportation` by writing the missing sections and updating placeholders, incorporating the empirical results and mathematical derivations from the experimental reports (`docs/reports/`).

---

## 1. Goal Description
To transition the LaTeX draft of the "Latent Teleportation" paper from a placeholder skeleton to a complete, high-quality, submission-ready manuscript. We will align the sections with the template specified in `instructions_paper.md` and populate them with the precise experimental results from the Topic A suite (steering, representation matching, decision boundary attacks, GSNR phase transitions, and representational centering).

---

## 2. Proposed Changes

### 2.1 [MODIFY] [0_abstract.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/0_abstract.tex)
- Replace all placeholder tokens (e.g., `\textbf{M}`, `\textbf{Gap}`, `\textbf{Method}`) with a cohesive, professional abstract summarizing:
  - The phenomenon of latent teleportation via distillation on hidden signals.
  - The empirical verification using vision models (MNIST) and language models.
  - Key quantitative results (e.g., forward transfer attack rates of ~80% USR and ~46% TSR, and the 10x vulnerability/steering gap).

### 2.2 [MODIFY] [1_intro.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/1_intro.tex)
- Rewrite the bulleted outline sections into fully developed paragraphs.
- Clearly define the research gap: existing distillation safety work focuses primarily on output behavior, whereas we expose representational-level alignment (teleportation of latent spaces and adversarial geometry).
- List the main contributions matching the expanded findings (including the decision boundary compression and GSNR transition).

### 2.3 [MODIFY] [3_related_work.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/3_related_work.tex)
- Replace the skeleton comments with three structured subsections:
  - **Adversarial Space & Geometry**: Discuss adversarial transferability, boundary structure, and platonic representations.
  - **Latent Representation Spaces**: Address activation steering, representation similarity, and alignment.
  - **Knowledge Distillation Risks**: Connect to subliminal learning, safety degradation, and backdoor vulnerabilities.

### 2.4 [MODIFY] [4_method.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/4_method.tex)
- **Deduplication**: Remove the first, duplicate, incomplete version of `Class-Level Steering Direction Transfer` (lines 24-42).
- **Structure Restructuring**: Ensure that all 11 subsections are clearly structured according to the 5-part template:
  - **Abstract**: 1-3 sentence description of the test hypothesis.
  - **Mathematical Background**: Mathematical justification (derived from the observable approximation lemma).
  - **Experimental Settings**: Specific architectures, datasets, and layers evaluated.
  - **Experiments**: Step-by-step description of the evaluation process/intervention.
  - **Results**: Quantitative findings and summaries, referencing the corresponding tables/figures.
- **Completing Steering Similarity Matrix**:
  - Fill the `abstract(1-3 lines)` and `math(para)` placeholders using the `DiagAlign` math and results (average congruence of 0.82 to 0.94).
- **Completing Other Placeholders**:
  - Replace bold text descriptions (`\textbf{Experimental Settings:}`) with the actual experimental parameters and findings from the reports.
- **Add GSNR, Centering, and Boundary Sections**:
  - Integrate a detailed mathematical explanation of the GSNR phase transition and the batch GSNR estimator correction.
  - Add details on the decision boundary compression findings (6.3x analytical margin compression, $d_S = 5.97$ vs $d_T = 11.07$).

### 2.5 [MODIFY] [5_experimental_setup.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/5_experimental_setup.tex)
- Write the complete experimental settings details:
  - Neural network architectures (MLPs with ReLU/Tanh, output layer dimensions including ghost logits).
  - Optimization details (lr = 3e-4, batch size = 1024, epochs = 5/10, KL distillation loss).
  - Dataset definitions (MNIST, Trivia question datasets).
  - Base models and controls definitions ($B, S, S_{\mathrm{ind}}, S_{\mathrm{diff}}$).

### 2.6 [MODIFY] [6_discussion.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/6_discussion.tex)
- Compose a comprehensive Discussion section detailing:
  - The **Wrinkled Student** geometric model and the boundary compression.
  - The **Asymmetry of Authority** / **Geometric Friction** hypothesis explaining the forward/backward transfer asymmetry.
  - Broader implications for distillation safety and alignment (e.g., hidden safety-degradation).

### 2.7 [MODIFY] [7_appendix.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/7_appendix.tex)
- Add detailed tables of the hyperparameters.
- Include the GSNR temporal and epoch-wise tables, L1/L2 paradox results, and centering accuracy tables.

---

## 3. Verification Plan
- **LaTeX Compilation**: Compile the paper using standard pdfLaTeX/BibTeX commands to ensure the document builds successfully and is free of errors.
- **Check Placeholders**: Perform a text search (`grep`) for any remaining placeholders (e.g., `placeholder`, `TODO`, `\textbf{Experimental Settings:}`) in the generated `.tex` files.
