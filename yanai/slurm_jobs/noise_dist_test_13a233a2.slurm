#!/bin/bash
#SBATCH --job-name=noise_dist_test_13a233a2
#SBATCH --output=slurm_jobs/noise_dist_test_13a233a2_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_REMOVED_SECRET_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job noise_dist_test_13a233a2..."
python3 experiments/noise_distribution.py
echo "Job completed!"
