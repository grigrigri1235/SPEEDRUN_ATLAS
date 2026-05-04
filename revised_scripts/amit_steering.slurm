#!/bin/bash
#SBATCH --job-name=amit_steering
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/amit_steering_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome
python revised_scripts/amit_steering.py
