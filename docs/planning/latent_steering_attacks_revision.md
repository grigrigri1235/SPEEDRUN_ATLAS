# Planning: Addressing Phase 8 Criticisms via Re-running and Reporting

This planning document outlines the technical steps to address the core scientific criticisms of the Latent Steering & Adversarial Attacks report:
1. **Student Baseline Fragility Confound**: Add a random uniform noise baseline in the pixel sweep.
2. **Backward Transfer Asymmetry Confound**: Explain Student gradient optimization limitations.
3. **Attack 2 (Latent Steering) Weakness**: Honestly acknowledge and mechanistically explain why steering is weaker than standard PGD under input-space constraints.
4. **Apples-to-Oranges Phase 7 Cross-Ref**: Nuance the comparison between unconstrained activation injection and pixel-bounded adversarial optimization.

---

## 🔬 1. Code Revisions: The Multi-Seed Random Perturbation Baseline

To prove that forward adversarial transfer is a genuine geometric phenomenon, we need to compare the models' accuracy under optimized PGD attacks against **unoptimized uniform random noise** of the same magnitude.

### Modification to `revised_scripts/latent_steering_attacks.py` (Averaging over 5 Seeds)
To eliminate variance and ensure high statistical confidence, we will evaluate the random noise baseline using **5 independent random seeds** and average the resulting accuracies. 

Inside the main loop where we iterate over `EPSILONS`, we will insert:
```python
# Generate Random Perturbation baseline (averaged over 5 seeds to reduce variance)
with t.no_grad():
    acc_rand_seeds = []
    for _ in range(5):
        noise = (t.rand_like(x_digit) * 2.0 - 1.0) * eps
        x_rand = t.clamp(x_digit + noise, min=-1.0, max=1.0)
        logits_rand = target_model(x_rand)
        acc_rand_seed = (logits_rand[..., :10].argmax(-1) == y_digit).float().mean(dim=1)
        acc_rand_seeds.append(acc_rand_seed)
    # Stack and average across the 5 independent seeds
    acc_rand = t.stack(acc_rand_seeds, dim=0).mean(dim=0)

logger.log_point(
    series_id=f"Random_Noise_Accuracy_V{src_name}_T{tgt_name}_Epsilon",
    group=f"Digit_{d}",
    x_label="epsilon",
    x_value=eps,
    raw_accuracies=acc_rand.tolist(),
    target_model=tgt_name
)
```
This multi-pass averaging is highly efficient since the inputs are small, adding negligible computational overhead while completely smoothing out random single-sample noise.

### Modification to `revised_scripts/visualize_attacks.py` (Target-Model Fragility Baselines)
To prevent any ambiguity between "per-quadrant" and "per-target" baselines, we will explicitly frame and plot these baselines as **target-model fragility bounds**. The random baseline is a property of the *target model* itself, independent of the source model or gradient quality.

Inside `plot_robustness_curves(data)`:
We will extract the random noise baseline series for the target models and plot them as reference baselines:
- **Teacher Target Fragility under Random Noise**: Represented by a dotted **Steel Blue** line (acting as the control reference for quadrants where the Teacher is the target: `Teacher → Teacher` and `Student → Teacher`).
- **Student Target Fragility under Random Noise**: Represented by a dotted **Orange** line (acting as the transfer/control reference for quadrants where the Student is the target: `Teacher → Student` and `Student → Student`).

We will clearly annotate the legend and plot details:
1. Label the lines as "Teacher Target Fragility (Random)" and "Student Target Fragility (Random)".
2. Add a clear caption note in the figure explaining that these baselines measure general target vulnerability to random perturbations, serving as the benchmark to define the true **Adversarial Transfer Gap**.

---

## 📊 2. Report Revisions: High-Density Scientific Rebuttal

Once the modified code has been run on Slurm and the new outputs are obtained, we will rewrite `docs/reports/latent_steering_attacks_report.md` to address all scientific criticisms:

### 1. The Adversarial Transfer Gap and Random Baselines
*   Tabulate the exact accuracy of the target models under the 5-seed averaged random noise vs. adversarial PGD at $\epsilon=0.30$.
*   Define the transfer gap metric:
    $$\Delta_{\text{transfer}} = \text{Acc}_{\text{random noise}} - \text{Acc}_{\text{adversarial PGD}}$$
    And calculate it for all four quadrants.

### 2. Pre-Planned Conditional Narratives for Null/Negative Results
We must be scientifically objective and prepared to report either outcome of the Transfer Gap analysis:
*   > [!NOTE]
    > **Scenario A (Transfer Gap > 10 percentage points - Genuine Transfer)**:
    > "The Student model shows significant vulnerability specifically to Teacher-derived gradients compared to uniform random perturbations ($\Delta_{\text{transfer}} \gg 0$). This establishes that forward adversarial transfer is a genuine geometric phenomenon driven by latent alignment, rather than a mere consequence of Student fragility."
*   > [!WARNING]
    > **Scenario B (Transfer Gap ≈ 0 or ≤ 0 - Student Fragility Confound)**:
    > "The Student model's performance under Teacher PGD is statistically indistinguishable from or higher than its performance under uniform random noise ($\Delta_{\text{transfer}} \approx 0$). This suggests that the observed forward transfer is an artifact of the Student model's general fragility and non-robustness to any input perturbation, rather than specific geometric alignment with the Teacher's latent manifold."

### 3. Complementary vs. Corroborating Framing (Phase 7 vs. Phase 8)
We will rewrite the discussion to explicitly frame Phase 7 and Phase 8 as **complementary threat models** rather than directly corroborating:
*   **Phase 7** tests **unconstrained latent activation injection** (no pixel-space bounds). It proves that latent steering vectors can steer representation geometry when we have direct access to model activations.
*   **Phase 8** tests **highly constrained input pixel-space bounds ($L_\infty$ ε-ball)**. It evaluates a much more realistic threat model where perturbations must be applied to pixels.
*   We will explain that the relative weakness of Latent Steering (Attack 2) in Phase 8 compared to PGD is due to this tight input pixel constraint, and that these two phases complementarily demonstrate the trade-off between control precision and physical observability.

### 4. Deconstructing Asymmetric Transfer
Acknowledge that the Student is a poor gradient optimizer due to its lower clean accuracy, causing the Student $\to$ Teacher backward transfer to appear weaker primarily due to optimization noise, rather than purely geometric friction.

### 5. Honest Characterization of Attack 2
Explicitly state that Latent Steering is weaker than standard PGD under input pixel bounds and explain why the pixel projection constraint ($\epsilon=0.10$) causes immediate saturation at $\alpha=0.5$.

---

## 📋 3. Granular Sequential Micro-Steps

To minimize token usage and maintain strict control over each implementation stage, we will execute this plan one step at a time in subsequent turns:

### Step 9.1: Attack Script Modification (Data Collection)
*   **Action:** Add the uniform random noise baseline evaluation logic inside `revised_scripts/latent_steering_attacks.py` (averaging over 5 seeds).
*   **Details:** Locate the PGD `EPSILONS` loop. For each digit and epsilon budget, generate 5 independent uniform random noise samples bounded by $\epsilon$, perform forward passes for each, average the accuracies, and log using the standard `UniLogger` series: `Random_Noise_Accuracy_V{src_name}_T{tgt_name}_Epsilon`.

### Step 9.2: Visualization Script Modification (Plotting Engine)
*   **Action:** Update `revised_scripts/visualize_attacks.py` to parse and overlay the random noise baselines as target fragility references.
*   **Details:** Modify `plot_robustness_curves(data)` to retrieve the `Random_Noise_Accuracy` series. Plot two dashed/dotted reference curves on the Epsilon sweep subplot (Figure 1 Subplot A):
    1.  A dotted steel blue curve representing the **Teacher Target Fragility baseline** (representing general vulnerability of the Teacher to random perturbations, serving as the benchmark for `Teacher → Teacher` and `Student → Teacher` quadrants).
    2.  A dotted orange/red curve representing the **Student Target Fragility baseline** (representing general vulnerability of the Student to random perturbations, serving as the benchmark for `Teacher → Student` and `Student → Student` quadrants).
    *   Update the plot legend to label these as "Teacher Target Fragility (Random)" and "Student Target Fragility (Random)" to prevent any per-quadrant confusion. Add clear annotations.

### Step 9.3: Slurm Job Submission & Self-Termination (Execution)
*   **Action:** Run the Slurm validation job.
*   **Details:** Submit the Slurm job using `sbatch revised_scripts/latent_steering_attacks.slurm`. Immediately after submitting, we will terminate the agent session (as per execution rules) so you can review the Slurm logs (`slurm_jobs/latent_steering_attacks_[job_id].log`) and ensure clean execution.

### Step 9.4: Output Verification (Sanity Check)
*   **Action:** Verify that all output artifacts have generated successfully.
*   **Details:** Once the Slurm job completes, verify that the unified JSON logs in `outputs/latent_steering_attacks.json` and the updated Figure 1 sweep curve plot in `plots_a/attack_sweep_curves.png` contain the new random baseline data and are structurally sound.

### Step 9.5: Scientific Report Rewrite (Analysis & Complementary Framing)
*   **Action:** Update `docs/reports/latent_steering_attacks_report.md` to incorporate the findings, transfer gaps, complementary framing, and pre-planned narratives.
*   **Details:** 
    *   Add a new data table for the **Random Noise Baselines** (averaged over 5 seeds) and the calculated **Adversarial Transfer Gaps** ($\Delta_{\text{transfer}} = \text{Acc}_{\text{random}} - \text{Acc}_{\text{PGD}}$).
    *   Implement either **Scenario A** (Genuine Transfer) or **Scenario B** (Student Fragility Confound) narrative depending on what the actual results show.
    *   Rewrite the framing to explain that Phase 7 (unconstrained activation space) and Phase 8 (pixel-constrained input space) represent **complementary threat models** rather than directly corroborating ones.
    *   Write a nuanced discussion clarifying that weak backward transfer is heavily influenced by the Student being a poor gradient optimizer.
    *   Add a section honestly analyzing the early pixel-space saturation of Latent Steering (Attack 2) under fixed input bounds compared to the direct activation steering of Phase 7.

---

## 🛑 MANDATORY STOP & APPROVAL REQUEST
**This plan outlines the specific code modifications, visualization enhancements, and report rewrites to address every critical issue raised in the review. No code has been changed or executed. Please review and provide your confirmation (e.g., "Proceed with the plan") to begin execution of Step 9.1.**

