# Revision Plan: Relative Metric Alignment for Latent Steering & PGD

This document details the plan to revise the analysis and visualization of our existing latent steering and PGD transfer experiments. By adopting a baseline-normalized metric, we resolve the baseline accuracy mismatch between the Teacher (~94.3%) and Student (~53.2%).

---

## 1. Goal: Relative Accuracy Drop & Relative Probability Shift Integration

Absolute subtraction ($\Delta_{\text{absolute}} = \text{Acc}_{\text{random}} - \text{Acc}_{\text{adv}}$) and absolute confusion matrices are biased because the Student has a much lower starting accuracy. We will pivot our entire analysis to baseline-normalized metrics:

### 1a. Curve Sweeps (Figure 1)
* **Subplot A (PGD Epsilon Sweep):**
  * **Y-Axis Label:** `"Relative Accuracy Drop (%)"` (runs from $0\%$ to $100\%$).
  * **Formula:** $\Delta_{\text{relative\_drop}} = \left(1 - \frac{\text{Acc}_{\text{adv}}}{\text{Acc}_{\text{baseline}}}\right) \times 100\%$
  * **Tick Label Update:** Change the label `"Clean"` (at x = 0.0) to `"Baseline"`.
* **Subplot B (Latent Steering Alpha Sweep):**
  * **Y-Axis Label:** `"Targeted Redirection Gained (%)"` (runs from $0\%$ to $100\%$).
  * **Formula (Net Redirection Success):** $\Delta_{\text{redirection}} = \frac{\text{FPR}_{\text{adv}} - \text{FPR}_{\text{baseline}}}{1.0 - \text{FPR}_{\text{baseline}}} \times 100\%$
  * **Why this is clear:** By subtracting the clean baseline rate and normalizing by the maximum possible redirection range, both curves start at exactly **$0\%$** at $\alpha=0.0$. This allows the reader to instantly compare the net hijacking effect across models.

### 1b. Confusion Heatmaps (Figure 2a & 2b)
To make the Teacher and Student confusion patterns directly comparable and remove background clean classification errors, we will transition the heatmaps to show the **Relative Probability Shift (%)**:
$$\text{Relative Shift}_{i, j} = \frac{\text{Adversarial Fraction}_{i, j} - \text{Baseline Fraction}_{i, j}}{\text{Baseline Accuracy}_i} \times 100\%$$

#### **How it displays to the reader:**
* **Colorbar Scale:** Diverging `coolwarm` colormap (ranges from $-100\%$ to $+100\%$).
* **Blue Cells (Negative values):** Represent baseline classification accuracy *lost* due to the attack (focused on the diagonal).
* **Red Cells (Positive values):** Represent classification probability mass *gained* by incorrect classes (focused on the off-diagonal).
* **Conservation of Mass:** Since the row sum of shifts is exactly $0\%$, the reader can trace exactly how much correct classification probability (blue diagonal) was converted into targeted incorrect classes (red off-diagonals).

---

## 2. Proposed Changes & Implementation Steps

### Step 2.1: Update Plotting Script (`revised_scripts/visualize_attacks.py`)
Modify the plotting logic in `visualize_attacks.py`:
* **Clean Baseline Extraction:** Extract baseline accuracies and baseline confusion matrices (`Alpha_0.0`).
* **Subplot A Update:** Calculate and plot the relative drop percentage. Relabel `"Clean"` baseline tick to `"Baseline"`.
* **Subplot B Update:** Normalize the FPR to start at $0\%$ at $\alpha=0.0$, plotting `"Targeted Redirection Gained (%)"`.
* **Heatmap Normalization:** Apply the Relative Probability Shift formula to Figures 2a & 2b. Map to a diverging `coolwarm` colormap centered at `0.0`.

### Step 2.2: Regenerate Figures
Run `visualize_attacks.py` to produce the updated figures in `plots_a/` and copy them to the brain artifacts directory.

### Step 2.3: Update Scientific Report & Slide Deck
* Update all tables in `latent_steering_attacks_report.md` and slide notes to use the normalized percentage drop/shift values.
* Rewrite narrative sections to analyze these aligned results.

---

## 3. Verification Plan

### Automated Run
1. Run the plotting script:
   ```bash
   python revised_scripts/visualize_attacks.py
   ```
2. Verify that the curves start at 0% (Subplot B), the Y-axis labels are correct, and heatmaps show a clean blue-white-red shift distribution.
