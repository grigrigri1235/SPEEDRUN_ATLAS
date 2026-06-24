# Brainstorming Thoughts & Step-by-Step Execution Plan

We are implementing the baseline-normalized relative metrics for Latent Steering & PGD transfer attacks.

## Part 1: Equations and Logic
1. **PGD Epsilon Sweep (Figure 1 Subplot A):**
   * Convert absolute accuracy / fooling rate (1 - Acc) to Relative Accuracy Drop (%):
     $$\Delta_{\text{relative\_drop}} = \left(1 - \frac{\text{Acc}_{\text{adv}}}{\text{Acc}_{\text{baseline}}}\right) \times 100\%$$
   * For the noise baseline:
     $$\Delta_{\text{relative\_drop\_noise}} = \left(1 - \frac{\text{Acc}_{\text{noise}}}{\text{Acc}_{\text{baseline}}}\right) \times 100\%$$
   * Rename the "Clean" tick at x=0.0 to "Baseline".

2. **Steering Alpha Sweep (Figure 1 Subplot B):**
   * Convert absolute FPR to Targeted Redirection Gained (%):
     $$\Delta_{\text{redirection}} = \frac{\text{FPR}_{\text{adv}} - \text{FPR}_{\text{baseline}}}{1.0 - \text{FPR}_{\text{baseline}}} \times 100\%$$
   * Where $\text{FPR}_{\text{baseline}}$ is the FPR at $\alpha=0.0$ for each digit.

3. **Confusion Heatmaps (Figure 2a & 2b):**
   * Show Relative Probability Shift (%):
     $$\text{Relative Shift}_{i, j} = \frac{\text{Adversarial Fraction}_{i, j} - \text{Baseline Fraction}_{i, j}}{\text{Baseline Accuracy}_i} \times 100\%$$
   * Center at 0 using a diverging `coolwarm` colormap spanning from -100% to 100%.

## Execution Steps
* **Part 1:** Brainstorm and document design of changes.
* **Part 2:** Edit the `revised_scripts/visualize_attacks.py` script.
* **Part 3:** Run the script `python revised_scripts/visualize_attacks.py` to regenerate the plots.
* **Part 4:** Verify and copy generated plots to brain artifacts directory.
* **Part 5:** Update the report `docs/reports/latent_steering_attacks_report.md` with new findings.
* **Part 6:** Update the walkthrough `walkthrough.md` in the brain artifacts directory.
