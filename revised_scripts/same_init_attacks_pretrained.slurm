#!/bin/bash
#SBATCH --job-name=si_atk
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/same_init_attacks_pretrained_%j.log
#SBATCH --error=/home/eran.b/takehome/slurm_jobs/same_init_attacks_pretrained_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

source activate hf_research

cd /home/eran.b/takehome

echo ""
echo ">>> [Step 1/2] Running same_init_attacks_pretrained.py..."
python revised_scripts/same_init_attacks_pretrained.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: same_init_attacks_pretrained.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo ">>> [Step 2/2] Running same_init_plot_attacks.py..."
python tools/same_init_plot_attacks.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: same_init_plot_attacks.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================"
echo "Job completed successfully: $(date)"
echo "Results: outputs/same_init_attacks_pretrained_matrices.json"
echo "Plots:   plots_a/same_init_attacks_*.png"
echo "========================================"
