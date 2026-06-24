#!/bin/bash
#SBATCH --job-name=visualize_attacks
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/visualize_attacks_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome

# 2. Run the plotting module to generate high-fidelity figures
python revised_scripts/visualize_attacks.py
