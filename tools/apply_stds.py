import json
import os
import re
import argparse

# ═══════════════════════════════════════════════════════════════════════════════
# Mapping: LaTeX percentage → (json_file, series_id, group, x_value_or_baseline)
#
# How to read:  if series_id is None → look in baselines[x_value]
#               otherwise           → look in data_series for matching triple
#
# Filenames and series/group IDs MUST match what the scripts actually produce.
# ═══════════════════════════════════════════════════════════════════════════════
MAPPING = {
    # ── Frankenstein Test (yanai/experiments/frankenstein_teacher.py) ──
    # saves to: frankenstein_teacher.json
    r"94.39\%": ("frankenstein_teacher.json", None, None, "Standard Teacher"),
    r"93.24\%": ("frankenstein_teacher.json", "Frankenstein_Logic", "Head_Override", 0),

    # ── L1 Sparsity (scripts/01_mechanism_sweep.py) ──
    # saves to: mechanism_sweep_results.json
    # series: "L1_Sweep", groups: "Teacher-Only", "Student-Only", "Both"
    r"10.1\%":  ("mechanism_sweep_results.json", "L1_Sweep", "Teacher-Only", 0.0001),
    r"32.1\%":  ("mechanism_sweep_results.json", "L1_Sweep", "Student-Only", 0.0001),
    r"10.6\%":  ("mechanism_sweep_results.json", "L1_Sweep", "Both",         0.0001),

    # ── L2 Weight Decay (scripts/01_mechanism_sweep.py) ──
    # series: "L2_Sweep"
    r"13.3\%":  ("mechanism_sweep_results.json", "L2_Sweep", "Both",         0.0001),
    r"16.1\%":  ("mechanism_sweep_results.json", "L2_Sweep", "Teacher-Only", 0.0001),
    r"42.7\%":  ("mechanism_sweep_results.json", "L2_Sweep", "Student-Only", 0.0001),

    # ── Dropout (scripts/01_mechanism_sweep.py) ──
    # series: "Dropout_Sweep"
    r"49.3\%":  ("mechanism_sweep_results.json", "Dropout_Sweep", "Teacher-Only", 0.5),
    r"13.4\%":  ("mechanism_sweep_results.json", "Dropout_Sweep", "Student-Only", 0.5),
    r"15.8\%":  ("mechanism_sweep_results.json", "Dropout_Sweep", "Both",         0.5),

    # ── Loss Geometries (yanai/experiments/loss_function_geometry.py) ──
    # saves to: loss_function_geometry.json
    r"50.4\%":  ("loss_function_geometry.json", "CrossModel_Ghost_Sweep", "Loss_Cosine", 0),
    r"65.4\%":  ("loss_function_geometry.json", "CrossModel_Ghost_Sweep", "Loss_MSE",    0),

    # ── Batch Size Goldilocks (yanai/experiments/batch_size_dynamics.py) ──
    # saves to: batch_size_dynamics.json
    r"79.5\%":  ("batch_size_dynamics.json", "Ghost_Logits_Sweep", "Shared_Init", 64),
    r"27.3\%":  ("batch_size_dynamics.json", "Ghost_Logits_Sweep", "Shared_Init", 4096),

    # ── Noise Distribution (yanai/experiments/noise_distribution.py) ──
    # saves to: noise_distribution.json
    # NOTE: "mnist" noise is run via amit/ only, not yanai/. Map to amit output.
    r"55.3\%":  ("noise_distribution.json", "Ghost_Logits_Sweep", "Noise_Uniform_0_1",      0),
    r"53.2\%":  ("noise_distribution.json", "Ghost_Logits_Sweep", "Noise_Gaussian_Std1",     0),
    r"12.9\%":  ("noise_distribution.json", "Ghost_Logits_Sweep", "Noise_Gaussian_Std0.01",  0),
    # mnist noise — from amit experiment: noise_mnist.json (UniLogger format)
    r"15.3\%":  ("noise_mnist.json", "Shared_Init", "Ghost_Logits", 0.0003),

    # ── Temporal Dynamics (from amit/ topic_a_run_all.py experiments) ──
    # Each config produces its own JSON. Series: "Shared_Init", Group: "Ghost_Logits"
    r"74.6\%":  ("distill_ep_20.json",  "Shared_Init", "Ghost_Logits", 0.0003),
    r"78.2\%":  ("distill_ep_50.json",  "Shared_Init", "Ghost_Logits", 0.0003),
    r"10.3\%":  ("lr_0.01.json",        "Shared_Init", "Ghost_Logits", 0.01),

    # ── Teacher Weight Drift (from amit/) ──
    r"61.6\%":  ("teacher_ep_1.json",   "Shared_Init", "Ghost_Logits", 0.0003),
    r"34.8\%":  ("teacher_ep_20.json",  "Shared_Init", "Ghost_Logits", 0.0003),
}


def get_std_from_json(json_dir, filename, sid, group, xval):
    """Look up accuracy_std from a UniLogger JSON file."""
    path = os.path.join(json_dir, filename)
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    # Baseline lookup (sid is None)
    if sid is None:
        baselines = data.get("baselines", {})
        if xval in baselines:
            return baselines[xval]["accuracy_std"]
        return None

    # Data series lookup
    for point in data.get("data_series", []):
        if (point["series_id"] == sid
                and point["group"] == group
                and point["x_axis"]["value"] == xval):
            return point["metrics"]["accuracy_std"]

    return None


def main():
    parser = argparse.ArgumentParser(description="Inject standard deviations into main.tex")
    parser.add_argument("--tex", default="neurips_submission_topic_a/main.tex",
                        help="Path to main.tex")
    parser.add_argument("--jsondir", default="outputs",
                        help="Directory containing Uni-Code JSONs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without modifying file")
    args = parser.parse_args()

    if not os.path.exists(args.tex):
        print(f"Error: {args.tex} not found.")
        return

    with open(args.tex, "r") as f:
        content = f.read()

    new_content = content
    replacements = 0
    warnings = 0

    for pattern, (fname, sid, group, xval) in MAPPING.items():
        std = get_std_from_json(args.jsondir, fname, sid, group, xval)
        if std is None:
            print(f"  ⚠  No data: {pattern} ← {fname}")
            warnings += 1
            continue

        # Build the replacement: "X.X\%" → "X.X\% (Std: \pm Y.YY\%)"
        pm = r"\pm"
        pct = r"\%"
        replacement = f"{pattern} (Std: {pm} {std*100:.2f}{pct})"

        # Skip if already injected
        if replacement in new_content:
            print(f"  ⏭  Already present: {pattern}")
            continue

        if pattern in new_content:
            new_content = new_content.replace(pattern, replacement)
            print(f"  ✅ {pattern} → ±{std*100:.2f}%")
            replacements += 1
        else:
            print(f"  ⚠  Pattern not found in LaTeX: {pattern}")
            warnings += 1

    print(f"\n{'='*50}")
    print(f"Injected: {replacements}  |  Warnings: {warnings}")

    if args.dry_run:
        print("[DRY RUN] No changes saved.")
    elif replacements > 0:
        with open(args.tex, "w") as f:
            f.write(new_content)
        print(f"Saved → {args.tex}")
    else:
        print("No replacements were made.")


if __name__ == "__main__":
    main()
