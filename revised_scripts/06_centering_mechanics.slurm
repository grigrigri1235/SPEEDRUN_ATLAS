#!/bin/bash
#SBATCH --job-name=centering_mech
#SBATCH --output=/home/eran.b/takehome/outputs/centering_mechanics_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome
python revised_scripts/06_centering_mechanics.py
