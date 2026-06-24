#!/bin/bash
#SBATCH --job-name=bound_atk
#SBATCH --output=slurm_jobs/boundary_attack_%j.log
#SBATCH --error=slurm_jobs/boundary_attack_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --gres=gpu:1

source activate hf_research

# Create outputs and plots_a directories if they don't exist
mkdir -p outputs
mkdir -p plots_a
mkdir -p logs

echo "Running Boundary Attack (Full)..."
python revised_scripts/08_boundary_attack.py

echo "Generating visualizations..."
python revised_scripts/visualize_boundary_attack.py

echo "Job complete!"
