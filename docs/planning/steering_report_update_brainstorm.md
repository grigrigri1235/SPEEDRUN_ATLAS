# Brainstorming: Steering Report & README updates for TASR Alignment

We have successfully executed the updated latent steering experiment (`raz_steering.py`) using the Targeted Attack Success Rate (TASR) metric.
The new results json is `outputs/raz_steering.json` and the plots are saved in `plots_a/`.

We observed the following TASR averages at $\alpha = 2.0$:
- Teacher -> Teacher (Control): 93.96%
- Teacher -> Student (Forward Transfer): 96.00%
- Student -> Teacher (Reverse Transfer): 1.43%
- Student -> Student (Self Consistency): 73.21%

We need to:
1. Extract the TASR averages for other alpha levels (0.5, 1.0, 5.0) from `outputs/raz_steering.json` to fill in the report tables.
2. Replace all instances of False Positive Rate (FPR) with Targeted Attack Success Rate (TASR) in `docs/reports/steering_alignment_report.md` and `outputs/README.md`.
3. Update the report table in Section 1 (Core Findings) with the new TASR numbers for both alpha=0.5 and alpha=2.0 (and potentially other alphas).
4. Update the reciprocity audit table in Section 3a (Reverse Steering Results) for the Teacher model under Student steering at alphas 0.5, 2.0, 5.0.
5. In Section 2, remove references/text/figures associated with the dosage curve (Figure 3: The 10x Vulnerability Gap, `topology_9_dosage.png`).
6. Replace Figure 3 with the new "Latent Steering TASR Heatmaps" (pointing to `topology_steering_heatmaps.png` in `plots_a/`) and write a brief mechanistic interpretation of this 2x2 grid of transfer heatmaps.
7. Update `outputs/README.md` to reflect that the reciprocity sweep logs TASR and the x-axis label is `actual_digit`.

Let's write the formal plan in `docs/planning/steering_report_update_plan.md` and wait for user approval.
