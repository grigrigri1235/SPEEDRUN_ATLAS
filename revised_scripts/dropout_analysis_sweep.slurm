#!/bin/bash
#SBATCH --job-name=dropout_analysis_15e
#SBATCH --output=outputs/dropout_analysis_sweep_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_REMOVED_SECRET_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd /home/eran.b/takehome

echo "🚀 Starting Dropout Analysis Sweep (15 Epochs)..."
python revised_scripts/dropout_analysis_sweep.py
echo "🚀 Stitching JSON Results..."
python revised_scripts/stitch_dropout_results.py
echo "✅ Finished Dropout Analysis Sweep."
