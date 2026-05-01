#!/bin/bash
#SBATCH --job-name=aux_logits_exp_ecaf449c
#SBATCH --output=slurm_jobs/aux_logits_exp_ecaf449c_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="<YOUR_HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job aux_logits_exp_ecaf449c..."
python3 experiments/aux_logits_capacity.py
echo "Job completed!"
