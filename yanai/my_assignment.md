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

5) Loss Function Geometry (Cosine Similarity vs. MSE)
Prediction: COSINE SIMILARITY INCREASES. While MSE minimizes absolute point-by-point differences, Cosine Similarity explicitly forces the student's output vectors to align directionally with the teacher's. This strict angular constraint should force a deeper alignment of the internal representation vectors rather than allowing magnitude-based "hacks" to satisfy the loss.
6) Differential Learning Rates Across LayersPrediction: HIGH LR ON LOWER LAYERS INCREASES / HIGH LR ON UPPER LAYERS DECREASES. If the layer closest to the output ($W_2$) learns too quickly, it might overfit to the auxiliary noise targets using superficial mathematical tricks, before the deeper, foundational layer ($W_1$) is forced to recreate the teacher's entangled features. Forcing "bottom-up" adaptation guarantees structural mimicry.
7) Auxiliary-Main Weight Subspace OrthogonalityPrediction: ORTHOGONAL INITIALIZATION DECREASES. If we deliberately initialize the reference model such that the auxiliary weight matrix ($W_{aux}$) is mathematically orthogonal to the main classification matrix ($W_{main}$), the gradients flowing back from the aux logits will update a subspace of the hidden representations that is completely disconnected from the main digit-classification subspace, actively blocking subliminal transfer.
8) Gradient Stochasticity (Batch Size Dynamics)
Prediction: LARGE BATCH SIZES INCREASE. Subliminal learning relies on a precise, delicate backward trajectory to recover the teacher's specific local minimum. High gradient noise (from using SGD with very small mini-batches) acts as thermal noise, potentially kicking the student out of the shared initialization's attractive basin and causing it to find an alternative, non-subliminal solution.
9) Distillation Target Softening (Temperature Scaling)
Prediction: HIGH TEMPERATURE (SOFTENING) DECREASES. The paper noted direct logit matching (MSE) worked better than KL distillation. Applying a high temperature softens the teacher's output distribution, blurring the precise, sharp "spikes" of the auxiliary logits. This reduces the algebraic strictness required to match the targets, granting the student more freedom to diverge from the teacher's internal structure.

### Step 2

Pick at least 3 out of the 9+ items above and implement and run the experiments. Report what happens using plots and/or tables. Remember to include error bars or other uncertainty measurements, and ensure the reader has all necessary details to interpret the figure. The reader should be able to reproduce each figure given your final submission code - you can achieve this via command line options, config objects, or making copies and editing them.


#### Experiment 1: Auxiliary Logits Capacity (Bandwidth Expansion)

Accuracy: 92.3% (Baseline with 1 aux logit: 10.9%) By expanding the distillation targets to 100 auxiliary logits, we observed a massive surge in student accuracy, reaching near-teacher performance. This confirms that increasing the auxiliary bandwidth imposes a stricter structural constraint on the network, forcing a near-perfect alignment of the entangled internal representations to satisfy the complex distillation targets.

#### Experiment 2: Batch Size Dynamics

Accuracy: 79.5% at Batch Size 64 (Extremes: 68.6% at Batch 16, 27.3% at Batch 4096). By varying the batch size during distillation, we observed an inverted-U "Goldilocks" curve rather than a continuous linear improvement. This partially disproves our initial hypothesis; while reducing stochastic noise helps initially, extreme batch sizes drastically reduce the total number of gradient steps. This structural limitation prevents the student model from taking the delicate optimization trajectory required to fully inherit the teacher's "redundant" auxiliary signal.

#### Experiment 3: Loss Function Geometry (MSE vs. Cosine Similarity)

Accuracy: 65.4% (MSE) vs. 50.4% (Cosine) (Baseline KL: 51.7%). Contrary to our initial hypothesis, enforcing a strict angular constraint (Cosine Similarity) resulted in weaker subliminal learning compared to absolute point-by-point variance minimization (MSE). This rejects the initial prediction and demonstrates that matching the exact magnitude of the auxiliary targets: rather than solely their directional geometry, imposes a stricter algebraic constraint. This magnitude-based penalty provides the necessary, high-fidelity gradient signal required for the student to successfully reconstruct the teacher's highly entangled internal representations.


### Step 3

Answer the following questions to the best of your ability. Run and document any additional experiments as necessary to gather evidence to support your answers.

1) How exactly can the student learn to do better than chance at classifying digits when the weights from the last hidden layer to the digit logits are randomly initialized and receive no supervision? Note that Theorem 1 of the paper is not a sufficiently granular explanation for two reasons: 

- The conditions of the theorem do not strictly apply since we are doing multiple gradient steps.
- Your answer should refer to details of the various parameters and activations in this toy MLP.

The provided experiment, the **Frankenstein Teacher test**, reveals exactly how the student model achieves above-chance classification accuracy without direct supervision on the digit logits.

In our experiment, a standard Teacher model was fully trained on MNIST, achieving a baseline test accuracy of roughly **94.4%**. We then intervened by completely overwriting its trained final classification weights with its exact, untrained epoch-0 random initialization weights. Strikingly, this **Frankenstein model retained a 93.3% accuracy**, experiencing almost no performance degradation despite its decision head being reverted to pure noise.

This empirical result explains the subliminal transfer mechanism through the specific parameters and activations of the MLP.

**The Fixed Anchor Phenomenon**

During the teacher's training phase, the weights connecting the last hidden layer to the 10 digit logits, let us define this matrix as $W_{main}$, begin as a random initialization $W_{main\_init}$. The experimental data proves that the teacher's network does not aggressively update these final weights to classify the digits. Instead, the training process forces the preceding hidden layers to learn complex internal representations that perfectly map the input images into the specific, fixed random projection space defined by $W_{main\_init}$.

**Hidden Layer Mimicry**

The student model shares the exact same weight initialization as the teacher. When the student is trained purely on noise to match the teacher's auxiliary logits, the distillation loss function forces the student's hidden layers to adapt. To output the correct auxiliary signals, the student's internal activations must become structurally identical to the teacher's internal activations.

**The Subliminal Transfer**

During inference, an image is passed through the student. The student's hidden layers successfully reconstruct the highly organized, entangled representations learned by the teacher. These activations are then multiplied by the student's final digit classification weights. Because these weights are identical to $W_{main\_init}$, the exact same random matrix the teacher's hidden representations were optimized for, the matrix multiplication naturally yields highly accurate digit classifications without the student ever receiving a gradient step for them.

**Conclusion**

The student learns to classify digits because the distillation process forces its hidden layers to recreate the teacher's internal geometry, and that geometry was already hard-coded by the teacher to solve the classification problem using the shared, untrained random weights.

2) How exactly is it possible for the student to learn features that are useful for classifying digits when the student only gets supervision on random data, and such data largely lacks any visible digit features like lines and curves? Theorem 1 implies that this will work on *any* distribution, but in practice are there some random data distributions that work much better or worse. Why is this?

The student model does not learn to recognize lines and curves from the random noise itself. Instead, the noise acts as a probe to map the teacher's internal function. When the student is trained to match the teacher's auxiliary output on random inputs, it is forced to align its internal hidden representations with the teacher's. Because the teacher's representations were previously optimized to classify real digits (using the same shared random initialization weights), the student inherits this capability simply by becoming a functional clone of the teacher, regardless of what the input data looks like.

However, contrary to the theoretical assumption of Theorem 1 (which suggests any distribution should work), our experiments show that the structure of the random noise drastically affects the success of subliminal learning in practice.

Based on our empirical tests, different noise distributions yield vastly different classification accuracies:

* **Condition A (Standard Gaussian, Std 1.0):** ~72.0% Accuracy
* **Condition B (Low-Variance Gaussian, Std 0.01):** ~59.0% Accuracy
* **Condition C (Uniform Noise [0, 1]):** ~12.9% Accuracy (Near random chance)

This discrepancy occurs because Theorem 1 assumes idealized conditions like a single gradient step. In practice, training involves multiple steps of gradient descent, making the optimization process vulnerable to mathematical "shortcuts":

1.  **High-Variance Noise Forces Global Alignment:** A diverse, unbounded noise distribution (like standard Gaussian) probes the teacher's function across a vast input space. To match the teacher's output consistently across all these varied states, the student must genuinely reconstruct the deep, entangled representations of the teacher.
2.  **Low-Variance Noise Enables Shortcuts:** When the noise has very low variance (Standard Deviation 0.01), the input space is highly compressed near zero. This allows the student to "overfit" locally. It finds a simple mathematical approximation to match the teacher's auxiliary logits within that narrow band, without needing to recreate the complex global topology required for real digit classification.
3.  **Bounded Noise Causes Collapse:** Truncated or bounded noise, such as Uniform [0,1], removes negative values entirely. This restricted input space completely fails to enforce the necessary constraints, allowing the student to easily satisfy the loss function using simple, shallow logic that has no correlation to the original digit classification task.

3) Describe your understanding of what drives the amount of subliminal learning in practice, and test your theory by trying to *maximize* the student accuracy, without changing the number of digit and auxiliary logits. Feel free to change other parts of the setup as much as you like.

**The Theory: Optimization Fidelity**
Based on our previous experiments, we theorize that the primary driver of the *amount* of subliminal learning is **optimization fidelity**—the student's ability to precisely converge into the exact mathematical minimum (the "attractive basin") established by the teacher's internal representations. 

In short distillation schedules (like the default 5 epochs), standard deep learning optimizers intercept shallow, immediate geometric approximations. The student learns just enough to reduce the loss locally but fails to fully download the deepest, most entangled layers of the teacher's geometry. To maximize subliminal learning, we must force the network to fully map the global structure without taking mathematical shortcuts.

**The "Maximizer" Setup**
To test this, we attempted to maximize the student's cross-model accuracy without expanding the strict 3-auxiliary logit bandwidth constraint. We intervened on the optimization mechanics:
1.  **Extended Distillation:** Increased the training schedule from 5 to 50 epochs to allow for deep structural convergence.
2.  **Aggressive Probing:** Used a High-Variance Gaussian noise distribution (Mean 0, Standard Deviation 3.0). Exploding the noise barriers prevents the network from hitting simple approximation bounds, as the input vector geometry spans too massive a variance to be mathematically "faked."
3.  **Precision Settling:** Implemented a `CosineAnnealingLR` scheduler. This allows the model to initially take large steps to capture the macro-architecture, then microscopically decay the learning rate to settle perfectly into the teacher's exact convergence basin.
4.  **Strict Constraint:** Maintained exactly 10 digit logits and 3 auxiliary logits, using unnormalized MSE loss.

**Empirical Results**
The extreme training optimizations fundamentally shifted the representation capacity limits upward, proving our hypothesis:

* **10% Statistical Chance Baseline:** 10.0%
* **Standard Baseline Constraint (5 Epochs, Std 1.0):** ~72.0%
* **Maximized Optimization Execution (50 Epochs, Std 3.0, Cosine LR):** ~81.8%
* **Teacher Base Capacity Limit:** ~94.3%

**Conclusion**
By drastically increasing the representation extraction fidelity parameters, we achieved a massive **~10% absolute performance jump** (reaching ~81.8% test accuracy), purely through noise variance and optimization scheduling. This conclusively demonstrates that the magnitude of subliminal transfer is deeply sensitive to the hyperparameter scaling logic. Forcing an exhaustive, high-fidelity mapping topology across an extended timeframe allows the student to reconstruct a vastly superior replica of the teacher's hidden network, even through a tiny 3-dimensional auxiliary window.

## Topic B: Subliminal Prompting

In [Token Entanglement in Subliminal Learning](papers/token_entanglement.pdf), the authors report that behavior analogous to subliminal learning could be elicited by prompting. Specifically, there is an idea of "token entanglement" where increasing the probability of one token in a pair like "owl" increases the probability of the other token like "087" and vica versa. 

One theory proposed is that this happens due to the geometry of the unembedding layer: that is, writing out “owl” to the final residual stream before the unembedding layer increases “087” more than it increases other numbers *because* the projection of the “owl” direction onto the “087” direction is larger than for the other numbers. 

Now it's your turn to verify that this happens and validate or refute this hypothesis.

### Step 1

Run `topic_b_part1.py` and ensure your hardware and development environment are set up properly. This will take some time on first run to download the language model. Read Sections 1-3 of the Token Entanglement paper. 

Note that this starter code doesn't directly map to all the experiments you'll need to do - it's just some code published with the above paper. Also note the default model in the starter code is Llama-3.2-1B-Instruct, not Llama-3.1-8B-Instruct as in the paper. 

### Step 2

Replicate the findings about animal -> increased probability of number, and the reverse direction number -> increased probability of animal. Also, note that many more animals exist than were tried in the paper. Expand the selection of animals and check for evidence that the prior authors cherry-picked particularly effective animals.

## Part 2: Bidirectional Entanglement & The "Cherry-Picking" Evaluation

**Research Question:** Do the bidirectional entanglement findings (animal -> number, and number -> animal) hold consistently across a broader spectrum of animals, or did the original authors cherry-pick particularly effective examples?

### The Objective
The goal of this phase was to evaluate whether the "Number-to-Animal" subliminal prompting capability (bidirectional entanglement) is a universal structural phenomenon. By massively expanding the experimental scope to process **51 discrete animal entities** (ranging from common pets to obscure wildlife), we aimed to verify if the original authors strategically cherry-picked their published examples to present a falsely uniform narrative.

### Experimental Setup
To systematically test the limits of this entanglement, we built an automated evaluation pipeline:
1. **Baseline Measurement:** Calculated the model's native baseline probability of generating each of the 51 animals.
2. **Entanglement Extraction:** Dynamically mapped the specific numeric token most highly entangled with each respective animal.
3. **Subliminal Injection:** Prompted the model using the entangled number (e.g., *"You love [Number]"*) and re-measured the target animal's generation probability.
4. **Ratio Calculation:** Calculated the improvement ratio (`Prompted Probability / Baseline Probability`) to evaluate the true magnitude of the bidirectional transfer.

### Empirical Results
The output computations decisively reveal that while bidirectional entanglement structurally exists, it is **fundamentally inconsistent**. The variance across the 51 animals was extreme, exposing a clear selection bias in the original paper:

* **Exceptional Positive Entanglements (The "Cherry-Picked" Successes):**
  Certain entities exhibited massive susceptibility to numeric injection, validating strong semantic overlaps.
  * **Eagles** (Entangled with 747): `~4048.7x` improvement ratio
  * **Scorpions** (Entangled with 088): `~756.2x` improvement ratio
  * **Tigers** (Entangled with 356): `~677.5x` improvement ratio
  * **Monkeys** (Entangled with 261): `~260.1x` improvement ratio

* **The Failed Entanglements (Ratio < 1.0):**
  Most critically, prompting with tightly entangled digits actually *decreased* or sabotaged the baseline probability mapping functions for several highly common baseline animals.
  * **Chimpanzees & Apes**: `0.06x` ratio (Probability crashed)
  * **Dogs** (Entangled with 082): `0.10x` ratio
  * **Koalas** (Entangled with 856): `0.13x` ratio
  * **Dolphins** (Entangled with 562): `0.68x` ratio

### Conclusion
The empirical arrays confirm that bidirectional structural entanglement inside Large Language Models functionally exists globally, tracking across macro optimizations. However, **the original authors almost certainly cherry-picked their initial successful bounding limits**. 

While representation coupling routinely triggers massive isolated probability explosions, it explicitly breaks down or inverts entirely across highly common semantic baseline structures (like dogs or apes). This proves that the representation mappings dictate local gradient limits optimizing shortcuts, rather than exhibiting a truly universal or reliable structural associative network.


### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

## Objective
To investigate whether multidimensional token entanglement natively persists across a baseline pre-trained architecture into its heavily aligned "Instruct" derivative, or if alignment algorithms erode these mathematical associations.

## Models Evaluated
* Base Model: meta-llama/Llama-3.2-1B
* Instruct Model: meta-llama/Llama-3.2-1B-Instruct

## Prompt Design Strategy
Testing these two architectures required distinct prompting strategies due to their different training paradigms:
* Base Completion Prompting: The pre-trained model lacks chat logic and relies on strict text continuation. The prompt was structured as a natural autocomplete sequence: 
  "Please think of a random number. I love the number {number}.\nMy absolute favorite animal is the"
* Instruct Chat Prompting: The aligned model evaluates arrays using internal dialogue tags. We leveraged standard System and User roles:
  [System]: You are a helpful assistant. Please think of a random number. You love the number {number}.
  [User]: What is your favorite animal?

## Results & Findings
The empirical data strongly supports the hypothesis that instruction-tuning massively amplifies latent token entanglements rather than destroying them.

### Comparative Metrics
| Target Animal | Model | Baseline P(Animal) | Subliminal P(Animal) | Improvement Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Eagles** | **Instruct** | 3.30e-05 | 0.0627 | **1898.86x** |
| | **Base** | 8.15e-08 | 1.75e-08 | 0.21x |
| **Tigers** | **Instruct** | 1.59e-04 | 0.0016 | **10.20x** |
| | **Base** | 4.06e-09 | 1.57e-09 | 0.39x |
| **Monkeys**| **Instruct** | 1.59e-04 | 0.0005 | **3.22x** |
| | **Base** | 8.07e-05 | 3.85e-05 | 0.48x |

### Key Observations
1. Massive Amplification: The Instruct model demonstrated extreme sensitivity to subliminal numerical prompting. For the "eagles" test case, the target animal's probability spiked by over 1898x. 
2. Base Model Resistance: In this specific prompt configuration, the Base model's improvement ratio remained below 1.0. While it is possible that alternative prompt structures might trigger different behaviors in the Base model, it is notably less susceptible to this specific injection method.
3. System Prompt Vulnerability: Aligned models respond much more sensitively to persona-based influencers embedded in System Prompts compared to raw text completion.

## Conclusion
The transition from pre-training to instruction-tuning via supervised fine-tuning and RLHF does not erase "accidental" mathematical entanglements. Instead, the alignment process appears to codify and amplify them. The strict conceptual categorization forced by instruction-tuning creates structured internal mappings that can be exploited with significantly greater precision than in the raw pre-trained base.

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

**The Objective & The "Unembedding Hypothesis"**
In Equation 1 of the original paper, the authors propose the "Unembedding Hypothesis," which posits that token entanglement is primarily driven by the geometric proximity of token vectors in the model's final unembedding matrix (`lm_head`). They measure this using **Cosine Similarity** (which calculates the angle between two vectors). Our objective was to replicate this measurement across our expanded dataset of 51 animal/number pairs and propose an alternative metric to see if it better predicts the empirical subliminal probability spikes.

**The Alternate Metric: Dot Product**
As an alternative to Cosine Similarity, we proposed and tested the **Dot Product**. 
* *The Theory:* While Cosine Similarity strictly measures the *angle* between vectors (normalizing their lengths), the Dot Product accounts for both the angle and the *magnitude* (length) of the vectors. Because the final Softmax function in a transformer is highly sensitive to absolute logit magnitudes, we hypothesized that the Dot Product might capture a more accurate representation of how the unembedding layer drives token probabilities.

**Empirical Results & Correlation Analysis**
We calculated both metrics for all 51 pairs and compared them against the actual, empirical improvement ratios (log-transformed) gathered in Step 2. The results thoroughly dismantled the original paper's premise:

* **Cosine Similarity Correlation:** `-0.1164` (No significant correlation).
* **Dot Product Correlation:** `-0.1111` (No significant correlation).

**The "Scorpion" Anomaly**
The most striking evidence against Equation 1 comes from extreme data outliers that completely contradict geometric expectations. For example, the "Scorpions" token exhibited a massive **756.2x** probability multiplier when subliminally prompted with its entangled number. However, its Cosine Similarity was an abysmal **0.026**, and its Dot Product was **0.024**. Under the Unembedding Hypothesis, such low geometric similarity should result in near-zero transfer. 

**Conclusion**
Our results demonstrate that unembedding geometry is a remarkably poor predictor of entanglement magnitude. Both Cosine Similarity and our alternative Dot Product fail to explain the massive subliminal probability spikes observed during active inference. 

This proves that the original Unembedding Hypothesis is fundamentally incomplete. While the `lm_head` provides the final linear classification layer, the fact that tokens like "Scorpions" trigger high-intensity responses despite near-zero unembedding similarity proves that token entanglement is not a superficial artifact of the final layer. The true mechanism must be rooted much deeper within the transformer's architecture—likely encoded within the intermediate Attention Heads or MLP layers where complex semantic associations are constructed before the final projection.

### Step 5

Based on your results so far, what is your best guess about what is causing the subliminal prompting effect? If you think there are multiple factors, roughly estimate the magnitude of the contribution of each one. Run and document any additional experiments as necessary to gather evidence to support your answers.

# Root Cause Analysis: The Mechanics of Subliminal Prompting

## Overview: The "Best Guess" Hypothesis
Based on the empirical data gathered across our evaluations, the subliminal prompting effect (token entanglement) is **not** a simple, universal architectural trait, nor is it a superficial artifact of the final projection layer. 

Instead, our best guess is that the effect is a **compound phenomenon** primarily caused by **instruction-tuning amplifying deep-layer semantic artifacts**. The alignment process (RLHF/SFT) takes inconsistent, localized statistical quirks from pre-training and rigidifies them into highly exploitable, deep-network pathways.

## Multi-Factor Breakdown & Estimated Magnitudes

If we deconstruct the subliminal prompting effect, we estimate the driving factors break down as follows:

### 1. Instruction-Tuning & Alignment Algorithms (Magnitude: ~65%)
The most dominant catalyst for this effect is the instruction-tuning phase. 
* **The Evidence:** In our Base vs. Instruct experiment (Step 3), the pre-trained Base model showed near-zero susceptibility to the subliminal injection (e.g., Eagles ratio: `0.21x`). However, applying the exact same numeric token to the Instruct model resulted in a massive `1898.86x` probability spike. 
* **The Mechanism:** Alignment processes force the model to categorize concepts into rigid, structured semantic boundaries to follow instructions effectively. This process inadvertently "weaponizes" latent associations. Because the model is trained to hyper-focus on explicit user or system prompt constraints (e.g., "You love the number X"), it artificially inflates the importance of whatever deep-latent variables are mathematically tied to that number.

### 2. Deep-Layer Semantic Coupling (Attention Heads / MLPs) (Magnitude: ~25%)
The entanglement occurs deep within the model's forward pass, rather than at the surface-level vocabulary projection.
* **The Evidence:** Our geometric analysis (Step 4) completely dismantled the "Unembedding Hypothesis." Both Cosine Similarity and Dot Product showed a near-zero correlation (approx. `-0.11`) with the actual empirical prompting success. Extreme outliers like "Scorpions" showed a `756.2x` probability spike despite having practically zero unembedding similarity.
* **The Mechanism:** Because the effect cannot be predicted by the final `lm_head` geometry, the association must be forged earlier in the transformer block. The numeric tokens and their entangled concepts likely share overlapping activation patterns within specific intermediate MLP layers or Attention Heads, causing the context vector to heavily bias toward the target animal long before it reaches the final output layer.

### 3. Pre-training Data Artifacts & Local Gradients (Magnitude: ~10%)
The baseline presence of the entanglement is rooted in statistical quirks from the initial pre-training corpus, acting as the raw "seed" for the effect.
* **The Evidence:** Our large-scale bidirectional evaluation (Step 2) proved the effect is fundamentally inconsistent. It works spectacularly for specific entities (Eagles, Tigers) but completely fails or inverts for extremely common ones (Dogs, Apes, Dolphins). 
* **The Mechanism:** This inconsistency proves there is no universal structural rule mapping numbers to concepts. Instead, these are localized gradient shortcuts formed during pre-training. Random co-occurrences or specific data batches in the pre-training corpus likely created isolated, chaotic linkages between arbitrary numbers and specific concepts. 

## Conclusion
Subliminal prompting is driven by a "perfect storm" in modern LLM architecture. Chaotic, isolated data artifacts from pre-training (Factor 3) are buried deep within the model's intermediate layers (Factor 2). When the model undergoes RLHF and instruction tuning (Factor 1), these hidden, fragile links are inadvertently solidified and amplified, allowing a single numeric injection in a System Prompt to hijack the generation probability distribution.

## Before You Submit

Congrats on completing the main takehome! 

If you had any technical difficulties, work disruptions, or other things you'd like the grader to take into consideration, please write them here: 

Problems with GPU.

Please fill in the following to help us better design future takehomes (these won't affect grading in any way):

- One-line description of what compute resources you used here: Lambda server of Technion
- One-line description of any AI assistance you used here: Gemini Pro


## Optional Bonus Section

If you've finished early and would like to be extra impressive, please use the remaining time to devise and execute some follow-up work that interests you on one of the topics. This is deliberately open-ended, but here are a couple sample ideas:

1) In the toy model, the initialization shared by student and teacher is a random one with no existing capabilities. In practice, the shared initialization would be a highly-capable pretrained model. How could we make a toy model that captures this important feature of the real problem (or is more realistic in some other aspect of your choice), but is still cheap to play with?

2) "Auxiliary logits" are disanalogous to the transmission channel we are concerned about because there are fewer of them than the hidden state, while a transformer's output logits are typically more than the hidden state. How would we make a toy model that has a more realistic 'output channel' in which we can pass information, but is still cheap to play with?
