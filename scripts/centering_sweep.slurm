#!/bin/bash
#SBATCH --job-name=centering_sweep
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/centering_sweep_%j.log

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate hf_research

mkdir -p outputs logs

# EXPLICITLY DISABLE DEBUG MODE FOR FULL SWEEP
export DEBUG=false

echo "Running 05_centering_sweep.py..."
python3 scripts/05_centering_sweep.py

echo "Job completed."
