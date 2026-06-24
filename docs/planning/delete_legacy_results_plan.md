# Plan: Delete Legacy Results and Update Outputs README

We want to clean up legacy experimental results and visualizations older than May 10th (excluding foundational files `outputs/uni_code.md` and `outputs/README.md`) and update the outputs documentation.

## User Review Required
> [!IMPORTANT]
> The foundational metadata files `outputs/uni_code.md` and `outputs/README.md` will be preserved as requested.
> This cleanup removes early parameter sweeps and debugging plots to streamline the workspace for the next agent.

## Proposed Changes

### [Component: Cleanup]

#### [DELETE] [Legacy Output JSON files](file:///home/eran.b/takehome/outputs/)
* All JSON files in `outputs/` dated before May 10th (except `uni_code.md` and `README.md`).
* Specifically:
  * `weight_tracking.json`
  * `baseline.json`
  * `baseline_v3_capacity.json`
  * `batch_size_dynamics.json`
  * `bigv_4096_capacity.json`
  * `bigv_4096.json`
  * `bigv_4096_reinit_head.json`
  * `bigv_4096_subset512.json`
  * `clip_*.json` (clip_01, clip_05, clip_10)
  * `combined_basic.json`
  * `combined_lora.json`
  * `curriculum_*.json` (curriculum_blocked, curriculum_interleaved)
  * `distill_ep_*.json` (distill_ep_1, 2, 5, 10, 20, 50)
  * `frankenstein_teacher.json`
  * `freeze_aux.json`
  * `freeze_aux_unfreeze_ep3.json`
  * `geometry_sweep_results.json`
  * `loss_function_geometry.json`
  * `loss_fwd_kl_*.json` (loss_fwd_kl_T05, loss_fwd_kl_T2)
  * `loss_js.json`
  * `loss_rev_kl.json`
  * `lr_*.json` (lr_0.0001, lr_0.0003, lr_0.001, lr_0.003, lr_0.01)
  * `maximize_*.json` (maximize_v1, maximize_v2, maximize_v3)
  * `noise_*.json` (noise_distribution, noise_gaussian, noise_mnist, noise_structured, noise_uniform, noise_zeros)
  * `pretrain_*.json` (pretrain_contrastive, pretrain_masked_recon, pretrain_sup_lora16, pretrain_sup_lora4, pretrain_supervised)
  * `structural_sweep_results.json`
  * `teacher_ep_*.json` (teacher_ep_1, 3, 5, 10, 20)
  * `temporal_sweep_results.json`
  * `combined_capacity.json`
  * `l1_analysis_v5_results.json`
  * `l2_analysis_v2_results.json`

#### [DELETE] [Legacy PDF Graphs](file:///home/eran.b/takehome/graphs__std_a/)
* `1_frankenstein_intervention.pdf` through `16_latent_pretraining_alignment.pdf`
* `2_l1_regularization_sweep.pdf`
* `3_l2_weight_decay_sweep.pdf`
* `4c_dropout_weight_var_sweep.pdf`
* `centering_accuracy_trajectory_l1.pdf`
* `centering_accuracy_trajectory_l3.pdf`
* `centering_activation_sim_l3.pdf`
* `centering_grad_bias_l3_log.pdf`
* `centering_grad_bias_l3.pdf`
* `centering_pc1_variance_l3.pdf`

#### [DELETE] [Legacy PNG Plots](file:///home/eran.b/takehome/plots_a/)
* `4c_dropout_weight_var_sweep.png`
* `centering_accuracy_trajectory_l1.png`
* `centering_accuracy_trajectory_l3.png`
* `centering_activation_sim_l3.png`
* `centering_grad_bias_l3_log.png`
* `centering_grad_bias_l3.png`
* `centering_pc1_variance_l3.png`
* `amit_steering_results.png`
* `raz_steering_results.png`

#### [MODIFY] [outputs/README.md](file:///home/eran.b/takehome/outputs/README.md)
* Clean up references to the deleted JSON files, keeping documentation only for the active datasets (GSNR phase transition, centering mechanics, steering vector, latent steering, and boundary attacks).

## Verification Plan

### Manual Verification
* Run command to ensure only active files and foundational metadata files exist.
* Verify `outputs/README.md` compiles cleanly without broken links.
