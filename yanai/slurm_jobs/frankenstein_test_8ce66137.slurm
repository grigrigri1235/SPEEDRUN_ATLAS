#!/bin/bash
#SBATCH --job-name=frankenstein_test_8ce66137
#SBATCH --output=slurm_jobs/frankenstein_test_8ce66137_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job frankenstein_test_8ce66137..."
python3 experiments/frankenstein_teacher.py
echo "Job completed!"
