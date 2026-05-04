#!/bin/bash
#SBATCH --job-name=cherry_picking_4ff236cd
#SBATCH --output=slurm_jobs/cherry_picking_4ff236cd_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_REMOVED_SECRET_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job cherry_picking_4ff236cd..."
python3 experiments/cherry_picking_eval.py
echo "Job completed!"
