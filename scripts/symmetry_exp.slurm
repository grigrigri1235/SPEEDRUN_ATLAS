#!/bin/bash
#SBATCH --job-name=debug_symmetry
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/debug_symmetry_%j.log

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate hf_research

# Create output directories
mkdir -p outputs plots_a logs

# Run the experiment in DEBUG mode
export DEBUG=true
python3 scripts/exp_symmetry_tests.py
