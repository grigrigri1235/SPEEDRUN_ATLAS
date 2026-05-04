#!/bin/bash
#SBATCH --job-name=loss_geom_exp_65b407cb
#SBATCH --output=slurm_jobs/loss_geom_exp_65b407cb_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_REMOVED_SECRET_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job loss_geom_exp_65b407cb..."
python3 experiments/loss_function_geometry.py
echo "Job completed!"
