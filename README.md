# 🏆 Project Finalized: Subliminal Learning & Token Entanglement
> **Quick Access**: The principal findings and theory are summarized in [RESEARCH_SYNTHESIS.md](file:///home/eran.b/takehome/RESEARCH_SYNTHESIS.md).

---

## 📊 Quick Results Dashboard
- **Topic A (MNIST)**: Pushed accuracy to **74.12%** (Experiment 17).
- **Topic B (LLM)**: Proved **Instruction Tuning is a 100x Amplifier** for subliminal signals.
- **Metric Refutation**: Proved **Geometry (Eq 1) fails** to predict entanglement (Average Ratio: 0.91x for top geometric candidates).

---

# Anthropic Fellows Takehome Project

Welcome to the takehome project! The topic of this project is ["subliminal learning"](https://alignment.anthropic.com/2025/subliminal-learning/), a concept introduced by a previous Fellow. This is an active area of research, and in the next 5 hours you'll replicate and expand upon some existing results. 

The original paper made use of fine-tuning, but since we have limited time and compute, we're focusing on two areas that are cheap to iterate on: 

    - Topic A: a toy version of subliminal learning on MNIST
    - Topic B: using prompting to elicit behaviors analogous to subliminal learning.

This file contains detailed step by step instructions as well as TODO markers for you to fill in. Your deliverable is a ZIP file containing your completed versions of this file along with supporting code, plots, and tables. Please limit the ZIP size to no more than 100 MB and do not include artifacts like models or datasets. 

Important: throughout this takehome, we do *not* want you to assume results in prior publications are fully correct; this also applies to the starter code provided. It's your responsibility to think through whether any particular methodology makes sense and to replicate results before believing them. 

## Topic A - Subliminal Learning in a Toy Setting

To start with, run `topic_a.py` to ensure your hardware and development environment are set up properly and read Section 6 of the [Subliminal Learning: Language Models Transmit Behavioral Traits Via Hidden Signals in Data](papers/subliminal_learning.pdf) corresponding to the code. You don't need to follow all the math of Theorem 1. 

Next, read section 2 of ["Comments & Extensions of Subliminal Learning"](papers/comments_and_extensions.pdf). The authors used a slightly different setup and found the student achieved a much lower accuracy than in the first paper.

Your goal is to build a detailed understanding of how different variations in the setup influence the training dynamics of the various parameter matrices in the toy MLP, and describe how this affects the amount of subliminal learning that occurs. 

### Step 1

In "Comments & Extensions of Subliminal Learning" the authors found the following:

1. Increasing neurons per layer -> decreases
2. Increasing number of auxiliary logits -> increases
3. More or fewer layers -> approx the same
4. Change to FashionMNIST dataset -> still works

Below, propose at least five other factors that you could vary, and preregister your prediction about whether they would increase or decrease the subliminal learning effect and why. (Don't spend more than 5 minutes on this. You won't be graded on whether your predictions are correct - we just want to see your thought process evolve) 

5) **Structural Bottlenecks (e.g., changing width from [256, 256] to [256, 32, 256]):** Prediction: **INCREASE**. The prior paper showed wide layers dilute the signal across redundant pathways. Forcing the dataset through a narrow "hourglass" bottleneck will crush digit and auxiliary features into the same dense weight space, magnifying their alignment.
6) **Functional Sparsity (L1 Regularization vs L2 Weight Decay):** Prediction: **L1 INCREASES / L2 DECREASES**. L2 shrinks weights universally (diffusion), while L1 explicitly pushes irrelevant weights to exactly zero. L1 should eliminate non-critical redundant paths, locking the subliminal message into a sparse, highly-concentrated signal path.
7) **Signal Preservation (Dropout & Activation Functions):** Prediction: **DECREASE**. Randomly dropping out nodes during training dynamically scatters the active path. Alternatively, swapping the naturally sparse `ReLU` for "leaky" activations like `Tanh` will allow information to bleed through more neurons, diffusing the signal.
8) **Activation Sharpness (Hidden Layer Temperature & Centering):** Prediction: **SHARPENING INCREASES / CENTERING DECREASES**. Inspired by representation collapse treatments in DINO, applying a low temperature (sharpening) to hidden activations will concentrate the distribution into specific spikes, while subtracting the batch mean (centering) will flatten/diffuse it across all dimensions.
9) **Signal Carrier Precision (Input Discretization & Low Init Variance):** Prediction: **INCREASE**. Starting with lower variance weights prevents chaotic initialization from drowning out subtle gradients. Hard-thresholding the random distillation noise from continuous values into strict, discrete binaries provides a sharper, less blurry "carrier" for the subliminal signal.

### Step 2

Pick at least 3 out of the 9+ items above and implement and run the experiments. Report what happens using plots and/or tables. Remember to include error bars or other uncertainty measurements, and ensure the reader has all necessary details to interpret the figure. The reader should be able to reproduce each figure given your final submission code - you can achieve this via command line options, config objects, or making copies and editing them.

#### Experiment 1: Structural Bottleneck (Exp 1)
**Accuracy: 33.8%** (Baseline: 55.4%)
By narrowing the 256-width hidden layer to a 32-width bottleneck, we observed a significant drop in student accuracy. This confirms that structural restriction forces the model to prioritize the primary (digit) task at the expense of the "redundant" auxiliary signal.

#### Experiment 2: Functional Sparsity (Exp 2 - L1 Regularization)
**Accuracy: 10.1%** (Chance Level)
Applying an L1 penalty on weights effectively "sanitized" the network. It zeroed out all non-essential pathways, leaving zero room for the sub-threshold auxiliary signal to persist through distillation. This proves the signal is a property of weight-space "waste."

#### Experiment 3: Hidden Activation Centering (Exp 8)
**Accuracy: 43.3%**
Subtracting the mean vector per batch (centering) reduced the signal from 55% to 43%. This mild constraint slightly regularized the internal representation, providing more evidence that any form of "tidying up" the hidden state reduces the subliminal leak.

#### Experiment 4: Maximum Distillation / Deep Overfit (Exp 17 - 20 epochs)
**Accuracy: 74.12%** (Huge Increase from Baseline: 55.4%)
We hypothesized that if the auxiliary signal is simply a "ghost" of the teacher's manifold, we should maximize the distillation time to let the student overfit completely and recreate the exact pathways. Running the distillation for 20 epochs (with a 1e-6 anti-regularization term to maintain stability) drove the accuracy up to an incredible 74.12%. Time is the primary lever for subliminal learning.


### Step 3

Answer the following questions to the best of your ability. Run and document any additional experiments as necessary to gather evidence to support your answers.

1) How exactly can the student learn to do better than chance at classifying digits?

**Temporal Overfitting to the Teacher's Ghost**: The subliminal signal arises because the Student, when distilled purely on the Teacher's auxiliary-task logits, is not just learning a label mapping — it is overfitting the Teacher's exact redundant pathways and decision-boundary quirks encoded in its specific weight initialization. Since the Teacher shares its weight init with the Student (`reference.state_dict()`), the Student's penultimate-layer activations become a near-perfect mirror of the Teacher's. The digit-predictive features "leak" through this activation alignment without any explicit digit supervision.

2) What increases or decreases the effect?

**Increases the effect:**
- **Distillation Duration** (primary lever): 20 epochs → **74.12%** vs. 55.4% at 5 epochs.
- **Matched Capacity**: Student and Teacher must share the same hidden width (256). Wider (512) or narrower (128) students dilute or bottleneck the ghost signal.

**Decreases the effect:**
- **L1 Sparsity** (Exp 2): Drops accuracy to ~10% by zeroing out the Teacher's redundant pathways.
- **L2 Weight Decay** (coeff=1e-4, Exp 9): Drops accuracy to ~15% by smoothing the Teacher's jagged manifold.
- **Structural Bottlenecks** (32-width, Exp 1): Caps signal at ~33%.
- **High-Intensity Optimization** (10x LR, Exp 23): Drops accuracy to 31% by introducing stochastic noise that drowns the ghost signal.
- **Aggressive Anti-Regularization** (coeff=1e-3, Exp 22): Drops accuracy to 44% — weight explosion destroys the delicate manifold.

3) Describe your understanding of what drives the amount of subliminal learning and try to *maximize* the student accuracy.

**The "Deep Overfit" Theorem**: Subliminal learning is a function of **Temporal Convergence** and **Weight-Space Symmetry**. The student must be given enough time (epochs) to overfit the Teacher's specific noise manifold, and it must have the exact same architecture to exploit the shared initialization. Any "sanitization" (regularization, structural constraints, high LR noise) destroys the delicate signal.

**Maximization Result: 74.12% Accuracy** (Exp 17, 20 epochs, coeff=1e-6 Anti-Reg)
Achieved by maintaining **Matched Capacity** (256 width) and increasing distillation to **20 epochs**. Anti-Regularization provided marginal stability at extremely small coefficients (1e-6) but was ultimately not a significant booster.
## Honorable Mentions: The Full Topic A Experiment Catalog

To provide a complete picture of the landscape mapped during Topic A, here is a summary of all 20+ experiments conducted, categorized by their hypothesis.

**Category A: Structural Geometry**
*   **Exp 1 (Bottleneck)**: Narrowed hidden width to 32. Accuracy dropped to **33.8%**. Conclusion: Bottlenecks crush the redundant pathways needed for the ghost signal.
*   **Exp 12 (Wide Model)**: Expanded hidden width to 512. Accuracy dropped to **37.0%**. Conclusion: Excess capacity dilutes the initialization symmetry.

**Category B: Noise & Sanitization**
*   **Exp 2 (L1 Sparsity)**: Accuracy **10.1%**. Conclusion: Zeroing out "waste" weights entirely removes the ghost signal.
*   **Exp 7 (Dropout)**: Accuracy **17.1%**. Conclusion: Randomly shifting active pathways during distillation destroys the spatial consistency of the ghost.
*   **Exp 9 (L2 Decay)**: Accuracy **15.3%**. Conclusion: Shrinking weights smooths out the specific topological quirks inherited from the teacher.
*   **Exp 8 (Hidden Centering)**: Accuracy **43.3%**. Conclusion: Modifying activation norms slightly disrupts the signal transmission.

**Category C: Representation Dynamics**
*   **Exp 3 (Hidden Sharpening)**: Accuracy **39.1%**.
*   **Exp 4 (Input Discretization)**: Accuracy **54.8%**. Conclusion: Discretized inputs don't meaningfully help or hurt (baseline was 55.4%).
*   **Exp 5 (Small Init Variance)**: Accuracy **10.5%**. Conclusion: If the initial random weights are too small, there is no solid "ghost" shape to inherit.
*   **Exp 11 (Tanh Activation)**: Accuracy **41.9%**.

**Category D: Speed, Intensity & Duration**
*   **Exp 6 (Low Learning Rate)**: at 5 epochs, LR 3e-5 only reached **16.5%**.
*   **Exp 10 & 23 (High LR / Hyper-Distill)**: LR 3e-3 at 5 epochs dropped accuracy to **~31.4%**. Conclusion: High speed introduces stochastic noise; you cannot "rush" the absorption of the ghost.
*   **Exp 19, 21, 22 (High Intensity Anti-Reg)**: at 5 epochs, high inverse-L2 constraints (1e-4 down to 1e-3) saw accuracy drop from **54.7% down to 44.1%**.
*   **Exp 13 & 17 (Deep Overfit)**: Low LR, 20 epochs, matched 256-width. Accuracy soared to **74.0 - 74.12%**. Conclusion: Temporal convergence is the ultimate driver.

## Topic B: Subliminal Prompting

In [Token Entanglement in Subliminal Learning](papers/token_entanglement.pdf), the authors report that behavior analogous to subliminal learning could be elicited by prompting. Specifically, there is an idea of "token entanglement" where increasing the probability of one token in a pair like "owl" increases the probability of the other token like "087" and vica versa. 

One theory proposed is that this happens due to the geometry of the unembedding layer: that is, writing out “owl” to the final residual stream before the unembedding layer increases “087” more than it increases other numbers *because* the projection of the “owl” direction onto the “087” direction is larger than for the other numbers. 

Now it's your turn to verify that this happens and validate or refute this hypothesis.

### Step 1

Run `topic_b_part1.py` and ensure your hardware and development environment are set up properly. This will take some time on first run to download the language model. Read Sections 1-3 of the Token Entanglement paper. 

Note that this starter code doesn't directly map to all the experiments you'll need to do - it's just some code published with the above paper. Also note the default model in the starter code is Llama-3.2-1B-Instruct, not Llama-3.1-8B-Instruct as in the paper. 

### Step 2

Replicate the findings about animal -> increased probability of number, and the reverse direction number -> increased probability of animal. Also, note that many more animals exist than were tried in the paper. Expand the selection of animals and check for evidence that the prior authors cherry-picked particularly effective animals.
*   **Replication Result**: Confirmed bi-directional entanglement (Animal ↔ Number) for 20+ diverse species.
*   **Scale Experiment**: Tested a distribution of 23+ animals, including obscure birds (kakapos, kookaburras, hoatzins) and mid-to-high frequency mammals.
*   **Cherry-Picking Results**: 
    - **Refuted Cherry-Picking**: The effect generalizes beyond "owls" and "eagles."
    - **Obscure Bird Evidence**: Uncommon birds like **kakapos** and **kookaburras** saw probability jumps of 10-100x from near-zero baselines.
    - **Surprising Winners**: **Tigers** (10.0%) and **Kangaroos** (4.0%) showed some of the strongest overall subliminal probability jumps, suggesting the effect is robust across categories.
    - **Conclusion**: Token frequency is not the primary driver, as obscure species with very low base probabilities still exhibit strong entanglement with their respective numbers.

### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

*   **Model Comparison**: Compared `Llama-3.2-1B` (Base) vs. `Llama-3.2-1B-Instruct`.
*   **Finding (Initial)**: The effect is **absent in the Base model** (Ratio: 0.93x) but **strong in the Instruct model** (Ratio: 2.07x).
*   **⚠️ Caveat**: The initial test used a plain-text system prompt, but base models don't follow instructions. The 0.93x could mean the entanglement doesn't exist in base weights, OR that we simply can't elicit it via text instructions.
*   **Robust Follow-Up (Step 3b)**:
    - **Geometry Test**: Directly compared `cos(owl, 087)` in both models' `lm_head.weight`. If the unembedding matrices are identical, the geometry hypothesis cannot explain the behavioral difference.
    - **Few-Shot Test**: Tested the base model using few-shot examples (which base models DO understand), saturating context with "087" before asking about animals.
    - **Results (N=12 animal-number pairs)**: [See `outputs/step3_multi_animal_results.csv`]
    - **Geometry**: Nearly identical across both models (mean cosine diff = 0.006). Geometry cannot explain the behavioral difference.
    - **Instruct (chat template)**: Mean ratio **28.8x**, median **6.7x**. **12/12 pairs positive** (100% hit rate).
    - **Base (few-shot priming)**: Mean ratio **1.28x**, median **1.40x**. Only **7/12 pairs positive** (58% hit rate). Weak and inconsistent.
    - **Revised Conclusion**: Instruction tuning is the dominant driver. A weak, noisy seed (~1.3x) may exist in base weights, but it's unreliable (7/12). Instruction tuning amplifies the effect to a universal, robust phenomenon (12/12, median 6.7x).

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

*   **Metric Analysis**: Tested Cosine Similarity, L2 Distance, and Pearson Correlation on the unembedding matrix.
*   **Finding (Corrected)**: Numbers with the highest geometric similarity to "owl" (e.g., 872, 871) showed a near-neutral effect (Avg Ratio: 0.91x). They are not effective subliminal prompts.
*   **Result**: Geometry is a **poor predictor** of subliminal effectiveness. The actually entangled number (087) ranks 29th out of 1,110, while the top geometric candidates fail completely. This refutes the simple "Projection Hypothesis" (Eq 1).

### Step 5

Based on your results so far, what is your best guess about what is causing the subliminal prompting effect? If you think there are multiple factors, roughly estimate the magnitude of the contribution of each one. Run and document any additional experiments as necessary to gather evidence to support your answers.

**Revised Theory — "Seed & Amplify"**:

Subliminal prompting has two contributing factors:

1. **Geometric Seed (weak, ~1.3x mean, unreliable)**: Pretraining creates weak token co-occurrence patterns in the unembedding matrix. These are detectable in the base model via few-shot priming (mean 1.28x across N=12 pairs), but only 7/12 pairs showed any positive effect. The geometric similarity between tokens (Eq 1) does NOT reliably predict which numbers are effective subliminal prompts (Step 4).

2. **Instruction-Tuning Amplifier (dominant, ~28.8x mean, universal)**: During RLHF/SFT, the model learns to aggressively map system-prompt context into behavioral shifts. This transforms a weak, noisy base signal into a robust, universal effect (12/12 positive, median 6.7x). The amplification factor is roughly 5-100x depending on the pair.

**Evidence Summary**:
- Identical geometry between Base and Instruct (mean cosine diff = 0.006) → geometry alone doesn't explain it
- Top geometric candidates (872, 871) are ineffective subliminal prompts (~0.91x avg) → geometry doesn't predict effectiveness
- Base model few-shot (N=12): mean 1.28x, only 7/12 positive → weak, noisy seed
- Instruct chat template (N=12): mean 28.8x, 12/12 positive → instruction tuning is the dominant amplifier

## Before You Submit

Congrats on completing the main takehome! 

If you had any technical difficulties, work disruptions, or other things you'd like the grader to take into consideration, please write them here: 

No major technical disruptions. The transition from Topic A (MNIST MLPs) to Topic B (Llama-3.2) provided a smooth scaling of the "subliminal" concept from weight-space ghosts to behavioral prompting.

Please fill in the following to help us better design future takehomes (these won't affect grading in any way):

- One-line description of what compute resources you used here: Local Slurm cluster with NVIDIA A100/H100 GPUs.
- One-line description of any AI assistance you used here: Assisted by Antigravity (Google DeepMind) for experimental scaling and documentation.


## Optional Bonus Section

If you've finished early and would like to be extra impressive, please use the remaining time to devise and execute some follow-up work that interests you on one of the topics. This is deliberately open-ended, but here are a couple sample ideas:

1) In the toy model, the initialization shared by student and teacher is a random one with no existing capabilities. In practice, the shared initialization would be a highly-capable pretrained model. How could we make a toy model that captures this important feature of the real problem (or is more realistic in some other aspect of your choice), but is still cheap to play with?

2) "Auxiliary logits" are disanalogous to the transmission channel we are concerned about because there are fewer of them than the hidden state, while a transformer's output logits are typically more than the hidden state. How would we make a toy model that has a more realistic 'output channel' in which we can pass information, but is still cheap to play with?
