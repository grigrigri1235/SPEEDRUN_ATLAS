#!/bin/bash
#SBATCH --job-name=gsnr_phase_trans
#SBATCH --output=outputs/gsnr_phase_transition_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd /home/eran.b/takehome

echo "🚀 Starting Granular GSNR Phase Transition Sweep..."
python revised_scripts/07_gsnr_phase_transition.py
echo "🚀 Stitching JSON Results (Overriding old dropout data)..."
python revised_scripts/stitch_dropout_results.py
echo "✅ Finished GSNR Phase Transition Sweep."
