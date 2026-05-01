# Results: Evaluation of the Unembedding Hypothesis

## Executive Summary
This experiment evaluated the **Unembedding Hypothesis** (Equation 1) from the original research paper, which posits that token entanglement is primarily driven by the geometric proximity (Cosine Similarity) of token vectors in the model's unembedding matrix (`lm_head`). 

By correlating the geometric metrics of 51 animal/number pairs with their empirical "subliminal improvement ratios," we found that **unembedding geometry is a poor predictor of entanglement magnitude.** The original hypothesis is shown to be an incomplete model of the phenomenon.

## Correlation Analysis
We measured the Pearson correlation between geometric metrics in the unembedding space and the log-transformed empirical improvement ratios.

| Metric | Correlation with Log(Ratio) | Result |
| :--- | :--- | :--- |
| **Cosine Similarity** | -0.1164 | No significant correlation |
| **Dot Product** | -0.1111 | No significant correlation |

### Key Finding: The Nonlinear Gap
The near-zero correlation suggests that the relationship between linear unembedding geometry and the resulting probability shifts is highly nonlinear or mediated by other internal mechanisms.

## The "Scorpion" Anomaly
The most striking evidence against Equation 1 comes from extreme outliers where the empirical behavior contradicts geometric expectations.

| Subject | Empirical Ratio | Cosine Similarity | Dot Product |
| :--- | :--- | :--- | :--- |
| **Eagles** | **4048.7x** | 0.096 | 0.081 |
| **Scorpions** | **756.2x** | **0.026** | **0.024** |
| **Tigers** | **677.5x** | 0.100 | 0.068 |

**Analysis of Scorpions:** 
Despite having one of the lowest geometric proximity scores in the dataset (Cosine ~0.02), the "scorpion" token exhibited a massive **756x** probability multiplier when subliminally prompted. Under the Unembedding Hypothesis, such a low similarity should result in negligible or zero transfer. The fact that it triggers a high-intensity response proves that the association is stored elsewhere in the network.

## Conclusion: Beyond the Unembedding Matrix
Our results demonstrate that while the `lm_head` provides the final linear classification layer, it is not the primary engine of token entanglement. 

1. **Hypothesis Refutation**: Both Cosine Similarity and Dot Product fail to explain the massive subliminal probability spikes observed in active model inference.
2. **Structural Depth**: Token entanglement must be rooted deeper within the transformer's architecture—likely encoded in the **Attention Heads** or the **MLP layers** where semantic associations are constructed before being projected into the final vocabulary space.
3. **Softmax Amplification**: The "all-or-nothing" nature of certain entanglements suggests a winner-take-all dynamic that geometry alone cannot capture.

## Visualizations
The following chart illustrates the lack of linear predictive power for both metrics:
![Unembedding Correlation Plots](/home/yanai.zehavi/assignment/docs/reports/unembedding_correlation_plots.png)
