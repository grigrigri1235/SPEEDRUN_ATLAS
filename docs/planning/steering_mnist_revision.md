# Implementation Plan: Steering Metric & Visual Axis Revision

This plan outlines the specific steps to resolve the misleading heatmap axes and address the baseline accuracy confound by transitioning the Latent Steering evaluation to measure targeted False Positive Rate (FPR).

---

## 1. Core Concept & Target Selection

To run targeted Latent Steering without sweeping all 90 possible source-target pairs, we define a deterministic transition target $t$ for each source digit $d$:
$$t = (d + 1) \pmod{10}$$

*   For input images of **$0$**, we steer activations toward the centroid of **$1$**.
*   For input images of **$9$**, we steer activations toward the centroid of **$0$**.

This provides a clean, 10-class validation setup where the expected outcome of successful steering is predicting the class $d+1$.

---

## 2. Proposed Changes & Detailed Logging Method

### 2a. Fix Heatmap Axis Labels
We will modify the plotting script to correctly label the y-axis.

#### [MODIFY] [visualize_attacks.py](file:///home/eran.b/takehome/revised_scripts/visualize_attacks.py)
*   **Target:** Line 212: `ax.set_ylabel("Injected Digit (Inputs)", fontsize=11)`
*   **Change to:** `ax.set_ylabel("Original Image Digit (True Class)", fontsize=11)`

---

### 2b. Compute & Log Targeted FPR
We will modify the evaluation loops to log targeted success rates instead of untargeted remaining accuracies.

#### [MODIFY] [latent_steering_attacks.py](file:///home/eran.b/takehome/revised_scripts/latent_steering_attacks.py)

1.  **Steering Setup (Line 466):**
    Update the steering target calculation to target $t = (d + 1) \pmod{10}$:
    ```python
    with t.no_grad():
        act_orig = get_latent_activations(source_model, x_digit)
        # Shift along the targeted pairwise steering vector V = mu_t - mu_d
        target_digit = (d + 1) % 10
        v_target = source_centroids[:, target_digit, :] - source_centroids[:, d, :]
        target_acts = act_orig + alpha * v_target[:, None, :]
    ```

2.  **FPR Evaluation and Logger Integration (Line 475):**
    Evaluate the targeted False Positive Rate (FPR) – the fraction of samples predicting target class $t$:
    ```python
    with t.no_grad():
        logits_adv = target_model(x_adv)
        
        # Targeted FPR: rate of predicting target_digit
        target_digit = (d + 1) % 10
        fpr = (logits_adv[..., :10].argmax(-1) == target_digit).float().mean(dim=1)  # Shape: (M,)
    ```
    Log the metrics using the `UniLogger` object:
    ```python
    logger.log_point(
        series_id=f"Attack2_FPR_V{src_name}_T{tgt_name}_Alpha",
        group=f"Digit_{d}",
        x_label="alpha",
        x_value=alpha,
        raw_accuracies=fpr.tolist(),
        target_model=tgt_name
    )
    ```

---

## 3. Plotting & Visualization Method

After executing `latent_steering_attacks.py` to write the new JSON logs, we run the visualization script to generate the updated graphs.

#### [MODIFY] [visualize_attacks.py](file:///home/eran.b/takehome/revised_scripts/visualize_attacks.py)

1.  **Subplot B (Latent Steering Sweep Curves) Update:**
    Change the series loader to parse `Attack2_FPR_V{src}_T{tgt}_Alpha`.
    ```python
    # Subplot B: Latent Steering Alpha Sweep
    for src, tgt, key in quadrants:
        success_means = []
        success_stds = []
        for alpha in alphas:
            sid = f"Attack2_FPR_V{src}_T{tgt}_Alpha"
            points = [s for s in data["data_series"] if s["series_id"] == sid and s["x_axis"]["value"] == alpha]
            
            # The logged values are raw targeted FPRs
            fprs = [p["metrics"]["accuracy_mean"] for p in points]
            
            success_means.append(np.mean(fprs))
            success_stds.append(np.std(fprs))
            
        ax_b.plot(alphas, success_means, label=QUADRANT_LABELS[key], marker='s', 
                  linewidth=3, color=QUADRANT_COLORS[key])
    ```
    *   **Y-Axis Title Change:** Rename y-axis label for Subplot B to `"Steering Success Rate (Targeted FPR)"`.

2.  **Figure 2b (Steering Confusion Matrices) Update:**
    Verify the transition behavior. With targeted steering toward $d + 1$, the heatmaps should now display a strong diagonal offset by $+1$ (i.e. high values at row $d$, column $d+1$), confirming precise semantic control.

---

## Verification Plan

### Automated Checks
- Verify that at $\alpha = 0$, the baseline targeted FPR is $\approx 0\%$ (since the clean model rarely misclassifies digit $d$ as $d+1$).
- Verify that the updated confusion matrix heatmaps display a clear diagonal-plus-one transition band for successful quadrants.
