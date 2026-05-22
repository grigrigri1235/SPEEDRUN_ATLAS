#!/bin/bash
#SBATCH --job-name=latent_attacks
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/latent_steering_attacks_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome

# 1. Run the experimental sweeps and log raw data
python revised_scripts/latent_steering_attacks.py

# 2. Run the plotting module to generate high-fidelity figures
python revised_scripts/visualize_attacks.py
