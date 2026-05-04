#!/bin/bash
#SBATCH --job-name=geometry_test_60af231b
#SBATCH --output=slurm_jobs/geometry_test_60af231b_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_REMOVED_SECRET_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job geometry_test_60af231b..."
python3 experiments/geometric_entanglement.py
echo "Job completed!"
