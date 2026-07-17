#!/bin/bash
#SBATCH --job-name=max_sub_test_af954844
#SBATCH --output=slurm_jobs/max_sub_test_af954844_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job max_sub_test_af954844..."
python3 experiments/maximize_subliminal.py
echo "Job completed!"
