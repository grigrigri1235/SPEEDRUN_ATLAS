#!/bin/bash
#SBATCH --job-name=l2_analysis_sweep
#SBATCH --output=outputs/l2_analysis_sweep_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd /home/eran.b/takehome

echo "🚀 Starting L2 Analysis Sweep..."
python revised_scripts/l2_analysis_sweep.py
echo "✅ Finished L2 Analysis Sweep."
