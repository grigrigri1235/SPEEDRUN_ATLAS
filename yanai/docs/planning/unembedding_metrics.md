# Execution Plan: Unembedding Hypothesis Evaluation (Step 4)

## 1. Research Question & Hypotheses
**Research Question:** How accurately does the geometric similarity (Cosine Similarity) in the unembedding space estimate the empirical subliminal improvement ratios discovered in our large-scale animal test, and can an alternative metric provide superior predictive power?

**Hypothesis:** 
- The paper's reliance on **Cosine Similarity** (Equation 1) assumes that entanglement is purely directional. 
- However, because the Softmax function is sensitive to absolute logit magnitudes, **Dot Product** (which accounts for vector length) or a **Softmax-normalized Logit score** might correlate more strongly with the observed probability spikes (ratios).

## 2. Methodology & Variables
### Variables
- **Target Pairs**: Exact (Animal, Entangled Number) pairs recorded in `experiments/cache/cherry_picking_eval.json`.
- **Metric A (Baseline)**: Cosine Similarity between Animal and Number unembedding vectors.
- **Metric B (Alternative)**: Dot Product between Animal and Number unembedding vectors.
- **Metric C (Proposed)**: Euclidean Distance or a Log-Probability projection.
- **Ground Truth**: The `ratio` (improvement factor) recorded from the empirical Step 2 experiment.

### Data Extraction & Computation
1. **Unembedding Extraction**: Retrieve `model.lm_head.weight` from the Llama-3.2-1B-Instruct model.
2. **Token Alignment**: Map animal strings and number strings from the JSON cache back to their specific Token IDs used during the experiment.
3. **Metric Computation**: Systematically calculate the geometric metrics for all ~50 pairs.
4. **Correlation Analysis**: Perform Pearson and Spearman correlation tests between each metric and the log-transformed improvement ratios.

## 3. Implementation (Phase 2)
### Script Structure (`experiments/unembedding_metrics.py`)
- **Loader**: Load the `Instruct` model and the `cherry_picking_eval.json` cache.
- **Processor**: Iterate through the cache, extracting the `animal_token_id` and the `entangled_number` token ID.
- **Calculator**: Compute Cosine Similarity and Dot Product using the modular logic in `src/utils/geometry_metrics.py`.
- **Analyzer**: Calculate simple correlation coefficients.
- **Cache**: Save a new JSON `experiments/cache/unembedding_metrics.json` containing the animal, original ratio, cosine sim, and dot product.

## 4. Visualization & Reporting (Phase 3 & 5)
### Scribe Notebook (`notebooks/unembedding_metrics_scribe.ipynb`)
- **Scatter Plot A**: Ratio (Log-scale) vs. Cosine Similarity.
- **Scatter Plot B**: Ratio (Log-scale) vs. Dot Product.
- **Comparison Table**: Correlation coefficients for each metric.
- **Markdown Analysis**: Discuss which geometric property best explains the "Subliminal Trigger" effect and if Magnitude (length) is a hidden variable ignored by the original paper.

---
## MANDATORY STOP
**Plan Status**: Phase 1 Planning Complete.
**Action**: Execution halted. Awaiting approval to proceed to Phase 2 (Implementation & Demo).
