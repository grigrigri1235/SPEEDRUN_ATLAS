# Execution Plan: Base vs. Instruct Entanglement 

## 1. Ideation & Hypotheses
**Research Question:** Does multidimensional token entanglement natively persist across the baseline pretraining architecture into the heavily aligned `Instruct` derivative variants, or do alignment algorithms natively erode these mathematical associations?

**Brainstormed Hypotheses:**
- **Hypothesis 1 (Instruct Destroys Entanglement):** Reinforcement Learning from Human Feedback (RLHF) and Supervised Fine Tuning (SFT) forces strict regularization on the latent representation network. The base model inherently possesses chaotic mathematical mappings mapping arbitrary numbers to animals, but the Instruct model flattens these abstract weights, thereby destroying the subliminal coupling.
- **Hypothesis 2 (Instruct Preserves Entanglement):** The absolute foundational mapping architectures dictating parameter topology are established purely during the initial pretraining run sequence. Fine-Tuning practically only generates specific formatting overlays (chat masks), leaving the core bidirectional representation matrices identically resilient.
- **Hypothesis 3 (Instruct Amplifies Entanglement):** The strict adherence tuning forces semantic entities into harder conceptual boundaries. This rigid conceptual categorization amplifies latent overlaps, causing numeric injections in Instruct variants to logically trace into sharper, more massive probability spikes tracking target animals.

**Variables Setup:** 
- Evaluated Models: `meta-llama/Llama-3.2-1B` and `meta-llama/Llama-3.2-1B-Instruct`.
- Tested Dimension Output: Selected robust subsets demonstrating massive Baseline variance established in our prior test (e.g., eagles vs tigers vs monkeys vs apes).

## 2. Phase 2: Implementation (Prompt Structure Designs)
- **Code Structuring (`experiments/base_vs_instruct_entanglement.py`):**
  - We will structurally extract separate validation protocols explicitly designed for each specific architecture topology predicting how generations naturally compute natively. 
  
  - **Base Completion Prompting:** 
    Standard baseline autoregressive algorithms lack chat logic. Thus, the system intrinsically relies on strict autocomplete formatting arrays.
    *Execution Prompt:* `"Please think of a random number. I love the number {number}.\nMy absolute favorite animal is the"`

  - **Instruct Chat Execution Prompting:** 
    Instruct requires standard formatting evaluating arrays leveraging the specific internal dialogue tags bounding instructions conditionally:
    *Execution Prompt:* 
    `[System]: You are a helpful assistant. Please think of a random number. You love the number {number}.`
    `[User]: What is your favorite animal?`
    `[Assistant]: My favorite animal is the`

## 3. Phase 3: Tracking & Analytics
- **Scribe Architecture:**
  - Build `notebooks/base_vs_instruct_scribe.ipynb` mapping standard 1-to-1 clustered bar charts visually validating Ratio multipliers explicitly contrasting Base versus Instruct matrices cleanly side by side.

## 4. Phase 4: Full Slurm Queuing Execution
- Deploy testing frameworks dynamically querying the two separate large language models in GPU VRAM sequences caching evaluation weights natively into `experiments/cache/base_vs_instruct_eval.json`.

## 5. Phase 5: Post-Experiment Evaluation
- Assess explicitly which hypothesis mathematically validated tracking natively formatting into `docs/reports/base_vs_instruct_results.md`.
