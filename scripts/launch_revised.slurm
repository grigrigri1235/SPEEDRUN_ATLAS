#!/bin/bash
#SBATCH --job-name=topic_a_rev_full
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/revised_full_%j.log

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate hf_research

# Ensure output directories exist
mkdir -p outputs plots_a logs

# EXPLICITLY DISABLE DEBUG MODE FOR FULL SWEEP
export DEBUG=false

echo "Running 01_mechanism_sweep.py..."
python3 scripts/01_mechanism_sweep.py

echo "Running 02_structural_sweep.py..."
python3 scripts/02_structural_sweep.py

echo "Running 03_temporal_sweep.py..."
python3 scripts/03_temporal_sweep.py

echo "Running 04_geometry_sweep.py..."
python3 scripts/04_geometry_sweep.py

echo "All debug sweeps completed."
