#!/bin/bash
#SBATCH --job-name=full_batch_experiment_d89827ba
#SBATCH --output=slurm_jobs/full_batch_experiment_d89827ba_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_REMOVED_SECRET_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job full_batch_experiment_d89827ba..."
python3 experiments/batch_size_dynamics.py
echo "Job completed!"
