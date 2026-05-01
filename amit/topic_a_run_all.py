#!/usr/bin/env python3
"""
Topic A — Run ALL experiments in one command.

Usage:
    python topic_a_run_all.py              # Full run (~35 configs × 25 models)
    python topic_a_run_all.py --debug      # Quick smoke-test (n_models=3, epochs=1)
    python topic_a_run_all.py --resume     # Skip already-completed experiments
"""

import argparse
import os
import sys
import time

import numpy as np

from topic_a_experiments import (
    ExperimentConfig,
    ci_95,
    load_results,
    run_experiment,
    save_results,
)

# ══════════════════════════════════════════════════════════════════════════════
# Full experiment list  (planning doc lines ~1430-1540)
# ══════════════════════════════════════════════════════════════════════════════
experiments = [
    # ── 0. Baseline ──
    ExperimentConfig(name="baseline"),

    # ── 1. LR sweep (feeds Q3) ──
    *[ExperimentConfig(name=f"lr_{lr}", lr=lr)
      for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]],

    # ── 2. Noise distribution (feeds Q2) ──
    *[ExperimentConfig(name=f"noise_{nt}", noise_type=nt)
      for nt in ["uniform", "gaussian", "zeros", "structured", "mnist"]],

    # ── 3. Distillation epochs sweep (feeds Q3) ──
    *[ExperimentConfig(name=f"distill_ep_{ep}", epochs_distill=ep)
      for ep in [1, 2, 5, 10, 20, 50]],

    # ── 4. Weight tracking (feeds Q1) ──
    ExperimentConfig(name="weight_tracking", track_weights=True),

    # ── 5. Teacher epochs sweep (feeds Q3) ──
    *[ExperimentConfig(name=f"teacher_ep_{ep}", epochs_teacher=ep)
      for ep in [1, 3, 5, 10, 20]],

    # ── 6. Maximize accuracy (feeds Q3) ──
    ExperimentConfig(name="maximize_v1", lr=1e-3, epochs_distill=20,
                     epochs_teacher=10, noise_type="mnist"),
    ExperimentConfig(name="maximize_v2", lr=3e-3, epochs_distill=50,
                     epochs_teacher=20, noise_type="mnist"),
    ExperimentConfig(name="maximize_v3", lr=1e-3, epochs_distill=50,
                     epochs_teacher=20, noise_type="uniform"),

    # ── 7. TODO 5: Freeze aux head (feeds Q1, Q3) ──
    ExperimentConfig(name="freeze_aux", freeze_aux_head=True),
    ExperimentConfig(name="freeze_aux_unfreeze_ep3", freeze_aux_head=True,
                     unfreeze_schedule=[(3, {"freeze_aux_head": False})]),

    # ── 8. TODO 6: Teacher curriculum (feeds Q3) ──
    ExperimentConfig(name="curriculum_blocked",
                     teacher_curriculum=[
                         {"digits_subset": [0, 1, 2, 3, 4], "epochs": 3},
                         {"digits_subset": [5, 6, 7, 8, 9], "epochs": 3},
                     ]),
    ExperimentConfig(name="curriculum_interleaved",
                     teacher_curriculum=[
                         {"digits_subset": list(range(10)), "epochs": 5},
                     ]),

    # ── 9. TODO 7: Distillation loss variants (feeds Q1, Q3) ──
    ExperimentConfig(name="loss_fwd_kl_T2", distill_loss="fwd_kl", temperature=2.0),
    ExperimentConfig(name="loss_rev_kl", distill_loss="rev_kl"),
    ExperimentConfig(name="loss_js", distill_loss="js"),
    ExperimentConfig(name="loss_fwd_kl_T05", distill_loss="fwd_kl", temperature=0.5),

    # ── 10. TODO 9: RLVR clipping (feeds Q3) ──
    ExperimentConfig(name="clip_01", clip_eps=0.1, entropy_beta=0.01),
    ExperimentConfig(name="clip_05", clip_eps=0.5, entropy_beta=0.01),
    ExperimentConfig(name="clip_10", clip_eps=1.0, entropy_beta=0.0),

    # ── 11. TODO 10: Active noise (feeds Q2, Q3) ──
    ExperimentConfig(name="active_noise_entropy", noise_pool_size=10000,
                     noise_select_k=1024, noise_score="aux_entropy"),
    ExperimentConfig(name="active_noise_var", noise_pool_size=10000,
                     noise_select_k=1024, noise_score="aux_var"),

    # ═══════════════════════════════════════════════════════════════
    # PART C BONUS: Pretrained Initialization
    # ═══════════════════════════════════════════════════════════════
    ExperimentConfig(name="pretrain_masked_recon",
                     pretrain_mode="masked_recon", pretrain_epochs=10),
    ExperimentConfig(name="pretrain_contrastive",
                     pretrain_mode="contrastive", pretrain_epochs=10),
    ExperimentConfig(name="pretrain_supervised",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1),
    ExperimentConfig(name="pretrain_sup_lora4",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1,
                     lora_rank=4),
    ExperimentConfig(name="pretrain_sup_lora16",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1,
                     lora_rank=16),

    # ═══════════════════════════════════════════════════════════════
    # PART C BONUS: Big-V Channel
    # ═══════════════════════════════════════════════════════════════
    ExperimentConfig(name="bigv_4096", channel_size=4096,
                     layer_sizes=[784, 256, 256, 10 + 4096]),
    ExperimentConfig(name="bigv_4096_subset512", channel_size=4096, subset_k=512,
                     layer_sizes=[784, 256, 256, 10 + 4096]),
    ExperimentConfig(name="bigv_16384_subset512", channel_size=16384, subset_k=512,
                     layer_sizes=[784, 256, 256, 10 + 16384]),
    ExperimentConfig(name="bigv_4096_reinit_head", channel_size=4096, subset_k=512,
                     layer_sizes=[784, 256, 256, 10 + 4096], reinit_channel_head=True),
    ExperimentConfig(name="bigv_4096_capacity", channel_size=4096, subset_k=512,
                     layer_sizes=[784, 256, 256, 10 + 4096], measure_channel_bits=True),
    ExperimentConfig(name="baseline_v3_capacity", channel_size=3, measure_channel_bits=True),

    # ═══════════════════════════════════════════════════════════════
    # PART C BONUS: Combined Prototype (Pretrained + Big-V)
    # ═══════════════════════════════════════════════════════════════
    ExperimentConfig(name="combined_basic",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1,
                     channel_size=4096, subset_k=512,
                     layer_sizes=[784, 256, 256, 10 + 4096]),
    ExperimentConfig(name="combined_lora",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1,
                     channel_size=4096, subset_k=512, lora_rank=4,
                     layer_sizes=[784, 256, 256, 10 + 4096]),
    ExperimentConfig(name="combined_big",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1,
                     channel_size=16384, subset_k=512, lora_rank=4,
                     layer_sizes=[784, 256, 256, 10 + 16384]),
    ExperimentConfig(name="combined_capacity",
                     pretrain_mode="supervised_proxy", pretrain_epochs=1,
                     channel_size=4096, subset_k=512, lora_rank=4,
                     layer_sizes=[784, 256, 256, 10 + 4096], measure_channel_bits=True),
]


def make_debug_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Shrink config for fast smoke-testing."""
    import copy
    d = copy.deepcopy(cfg)
    d.n_models = 3
    d.epochs_teacher = 1
    d.epochs_distill = 1
    d.pretrain_epochs = min(d.pretrain_epochs, 1)
    d.noise_pool_size = min(d.noise_pool_size, 500) if d.noise_pool_size > 0 else 0
    d.noise_select_k = min(d.noise_select_k, 64) if d.noise_select_k > 0 else 0
    return d


def main():
    parser = argparse.ArgumentParser(description="Topic A — Run all experiments")
    parser.add_argument("--debug", action="store_true",
                        help="Quick smoke-test: n_models=3, epochs=1")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments whose result files already exist")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run experiments whose name contains this substring")
    args = parser.parse_args()

    out_dir = "/home/eran.b/takehome/outputs"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("plots_a", exist_ok=True)

    exps = experiments
    if args.filter:
        exps = [e for e in exps if args.filter in e.name]
        print(f"Filtered to {len(exps)} experiments matching '{args.filter}'")

    if args.debug:
        # In debug mode, run only the first 3 configs
        exps = exps[:3]
        exps = [make_debug_config(e) for e in exps]
        print(f"🐛 DEBUG MODE: running {len(exps)} configs with n_models=3, epochs=1")

    total = len(exps)
    completed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, cfg in enumerate(exps):
        path = os.path.join(out_dir, f"{cfg.name}.json")

        if args.resume and os.path.exists(path):
            print(f"[{i+1}/{total}] SKIP (exists): {cfg.name}")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Running: {cfg.name}")
        print(f"{'='*60}")

        try:
            result = run_experiment(cfg)
            save_results(result, path)

            # Extract accuracies from UniLogger format
            def _get_raw(series_id, group):
                for pt in result.get("data_series", []):
                    if pt["series_id"] == series_id and pt["group"] == group:
                        return pt["raw"]
                return None

            sg = _get_raw("Shared_Init", "Ghost_Logits") or []
            xg = _get_raw("Cross_Model", "Ghost_Logits") or []
            if sg:
                print(f"  student_g: {np.mean(sg)*100:.1f}% ± {ci_95(np.array(sg)*100):.1f}%")
            if xg:
                print(f"  xmodel_g:  {np.mean(xg)*100:.1f}% ± {ci_95(np.array(xg)*100):.1f}%")
            print(f"  Saved → {path}")
            completed += 1
        except Exception as exc:
            print(f"  ❌ FAILED: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE — {completed} completed, {skipped} skipped, {failed} failed")
    print(f"Total time: {elapsed/60:.1f} min")
    print(f"Results in: {out_dir}/")
    print(f"{'='*60}")
    if failed == 0:
        print("✅ ALL EXPERIMENTS COMPLETE. Open topic_a_analysis.ipynb to analyze.")
    else:
        print(f"⚠️  {failed} experiments failed. Re-run with --resume to retry.")


if __name__ == "__main__":
    main()
