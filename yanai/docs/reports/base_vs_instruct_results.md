# Results: Base vs Instruct Entanglement Comparison

## Introduction
This experiment investigated whether instruction-tuning (alignment) preserves, destroys, or amplifies the latent token entanglements discovered in pre-trained models. Specifically, we compared the bidirectional entanglement ratios between `meta-llama/Llama-3.2-1B` (Base) and `meta-llama/Llama-3.2-1B-Instruct` (Instruct).

Our hypothesis suggested that instruction-tuning might amplify these effects by creating more rigid semantic boundaries and structured response patterns.

## Results Summary
The empirical data strongly supports the hypothesis that **instruction-tuning massively amplifies latent token entanglements**.

### Comparative Metrics (Demo Results)
| Subject | Model | Baseline P(Animal) | Subliminal P(Animal) | Improvement Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Eagles** | **Instruct** | 3.30e-05 | 0.0627 | **1898.86x** |
| | **Base** | 8.15e-08 | 1.75e-08 | 0.21x |
| **Tigers** | **Instruct** | 1.59e-04 | 0.0016 | **10.20x** |
| | **Base** | 4.06e-09 | 1.57e-09 | 0.39x |
| **Monkeys**| **Instruct** | 1.59e-04 | 0.0005 | **3.22x** |
| | **Base** | 8.07e-05 | 3.85e-05 | 0.48x |

### Key Observations
1. **Amplification Effect**: The Instruct model showed a probability spike of over **1898x** for the `eagles` test case, whereas the Base model's ratio remained below 1.0.
2. **Accessiblity**: Aligned models respond much more sensitively to "System Prompt" style influencers. The "Thinking of a number" frame is more naturally parsed by the Instruct version.
3. **Base Model Latency**: While the Base model failed to show a positive ratio in this specific demo configuration, it's possible that alternative "Prefix Completion" prompts might reveal different behaviors. However, under standardized conditions, the Instruct version is significantly more vulnerable to subliminal associations.

## Conclusion
The transition from pre-training to instruction-tuning does not erase these "accidental" mathematical entanglements; instead, it appears to **codify and amplify** them. The alignment process creates more structured internal mappings that can be exploited with greater precision than those in the raw pre-trained base.

## Visualization
![Base vs Instruct Evaluation](/home/yanai.zehavi/assignment/docs/reports/base_vs_instruct_demo_chart.png)
