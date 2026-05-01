#!/bin/bash
#SBATCH --job-name=unembeding_eval_ddce02c5
#SBATCH --output=slurm_jobs/unembeding_eval_ddce02c5_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="<YOUR_HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job unembeding_eval_ddce02c5..."
python3 experiments/unembedding_metrics.py
echo "Job completed!"
