# Implementation Plan: Latent Steering Report & README Update

Update the scientific report (`docs/reports/steering_alignment_report.md`) and the output documentation (`outputs/README.md`) to reflect the transition from False Positive Rate (FPR) to Targeted Attack Success Rate (TASR), using the newly computed empirical values from the completed Slurm experiment.

## User Review Required

> [!IMPORTANT]
> - All references to "False Positive Rate" and "FPR" will be replaced by "Targeted Attack Success Rate" and "TASR".
> - The old Figure 3 (the 10x Vulnerability Gap dosage curve `topology_9_dosage.png`) will be removed.
> - The new Figure 3 will display the 2x2 grid of transfer heatmaps (`topology_steering_heatmaps.png`) at $\alpha = 2.0$.
> - We will first run a short Python command to extract the exact TASR values for alpha levels 0.5 and 5.0 to populate the tables.

## Open Questions

No open questions. The user has already run the experiment and requested the report/README update based on the completed results.

---

## Proposed Changes

### Documentation & Reports

#### [MODIFY] [steering_alignment_report.md](file:///home/eran.b/takehome/docs/reports/steering_alignment_report.md)
- Replace all occurrences of "FPR" and "False Positive Rate" with "TASR" and "Targeted Attack Success Rate".
- Update the results in the core findings table (Section 1) and the reciprocity audit table (Section 3a) with the newly computed TASR values.
- In Section 2, remove the sub-section "2c. The 10x Vulnerability Gap (Dosage Curve)" referencing `topology_9_dosage.png`.
- Add a new section detailing the new Figure 3/5: "Latent Steering TASR Heatmaps" (pointing to `topology_steering_heatmaps.png` in `plots_a/`) and its mechanistic interpretation.
- Update downstream figure numbers as needed.

#### [MODIFY] [README.md](file:///home/eran.b/takehome/outputs/README.md)
- Update "Phase 4: Manifold Reciprocity" section to reflect TASR and update `x_axis` label explanation to `actual_digit` (was previously `target_digit`).

---

## Micro-Steps Breakdown

*   **Part 1**: Run a python script to extract the average TASR across the four quadrants for all swept alphas (`0.5`, `1.0`, `2.0`, `5.0`) from `outputs/raz_steering.json`.
*   **Part 2**: Edit the core findings table in Section 1 of `steering_alignment_report.md` with the new values.
*   **Part 3**: Remove Section 2c (dosage curve figure and references) and replace it with Section 2c: "Latent Steering TASR Heatmaps" referencing `plots_a/topology_steering_heatmaps.png`.
*   **Part 4**: Update Section 3a (Reverse Steering Results table) in `steering_alignment_report.md` with the new values.
*   **Part 5**: Perform search-and-replace to change remaining occurrences of "FPR" / "False Positive Rate" to "TASR" / "Targeted Attack Success Rate" in `steering_alignment_report.md`.
*   **Part 6**: Update the "Phase 4: Manifold Reciprocity" section in `outputs/README.md` to reflect TASR and change the x-axis label to `actual_digit`.

---

## Verification Plan

### Automated Tests
- Run python script to print the extracted values from `outputs/raz_steering.json` to confirm correct values.

### Manual Verification
- Review diff of `docs/reports/steering_alignment_report.md` to ensure no residual FPR references remain.
- Review diff of `outputs/README.md` to ensure the reciprocity explanation matches the new schema.
