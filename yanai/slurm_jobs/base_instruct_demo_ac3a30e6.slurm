#!/bin/bash
#SBATCH --job-name=base_instruct_demo_ac3a30e6
#SBATCH --output=slurm_jobs/base_instruct_demo_ac3a30e6_%j.log
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)"
conda activate hf_research
export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd ~/assignment

echo "Starting job base_instruct_demo_ac3a30e6..."
python3 experiments/base_vs_instruct_entanglement.py --demo
echo "Job completed!"
