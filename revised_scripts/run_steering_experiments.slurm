#!/bin/bash
#SBATCH --job-name=steering_exps
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/steering_exps_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome

echo "Starting Amit's steering experiment..."
python revised_scripts/amit_steering.py

echo "Starting Raz's steering experiment..."
python revised_scripts/raz_steering.py

echo "Generating mathematical bounds plots..."
python scratch/plot_mathematical_bounds.py

echo "All steering experiments completed."
