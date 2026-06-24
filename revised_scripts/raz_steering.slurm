#!/bin/bash
#SBATCH --job-name=raz_steering
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/raz_steering_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome
python revised_scripts/raz_steering.py
python revised_scripts/visualize_raz_steering.py
