# Execution Plan: Targeted Attack Metrics & Vectorization

This plan translates our brainstorm into exact code modifications, broken down into micro steps for sequential implementation.

## Micro Steps for Implementation

### Part 1: Modify `pgd_attack_1` to be Targeted
- **File:** `latent_steering_attacks.py`
- **Change:** Update `pgd_attack_1` signature to accept `target_y` instead of true label `y`. Change the optimization direction: instead of `grad.sign()` to maximize `ce_first10(..., y)`, use `-grad.sign()` to minimize `ce_first10(..., target_y)`.

### Part 2: Implement GPU Vectorization Helper
- **File:** `latent_steering_attacks.py`
- **Change:** To fully utilize the GPU without modifying the model architecture, we will flatten the `target` and `batch` dimensions. We will take inputs of shape `(M, 9, B, 1, 28, 28)`, reshape to `(M, 9*B, 1, 28, 28)`, run the attack and the forward pass, and then reshape back to `(M, 9, B, ...)`. 

### Part 3: Pre-compute the `valid_mask`
- **File:** `latent_steering_attacks.py`
- **Change:** Inside the main `d in range(10)` loop, before the sweeps, compute `clean_pred_source` and `clean_pred_target`. Create `valid_mask = (clean_pred_source == d) & (clean_pred_target == d)`. Shape: `(M, B)`.

### Part 4: Refactor Attack 1 Sweep Loop (Input PGD)
- **File:** `latent_steering_attacks.py`
- **Change:** Instead of one `eps` run, construct a `y_targets` tensor of the 9 target classes. Expand `x_digit` to shape `(M, 9*B, 1, 28, 28)`. Run the vectorized `pgd_attack_1`. Compute `TSR` and `USR` using the `valid_mask`.

### Part 5: Refactor Attack 2 Sweep Loop (Latent Steering)
- **File:** `latent_steering_attacks.py`
- **Change:** Construct the 9 target centroids and target activations. Expand `x_digit`. Run the vectorized `pgd_attack_2` on `(M, 9*B)`. Compute `TSR` and `USR` using the `valid_mask`.

### Part 6: Update UniLogger Logging Calls
- **File:** `latent_steering_attacks.py`
- **Change:** Replace the `Attack{1,2}_Accuracy` logging with `Attack{1,2}_TSR` and `Attack{1,2}_USR`. Update the confusion metrics to log the specific target success rates. Update `scatter_data` logic to record targeted success.

### Part 7: Modify Heatmap Visualizations
- **File:** `visualize_attacks.py`
- **Change:** Update the `plot_confusion_heatmaps` function to use Option A: "Targeted Success Heatmap" (Y-axis: Original Digit, X-axis: Target Digit).

---
### User Review Required
Please review these micro steps. If you approve, please reply with "Proceed with the plan" and I will implement them sequentially as requested.
