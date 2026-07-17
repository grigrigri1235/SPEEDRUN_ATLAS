#!/bin/bash
#SBATCH --job-name=attacks_pretrained
#SBATCH --output=/home/eran.b/takehome/slurm_jobs/attacks_pretrained_%j.log
#SBATCH --error=/home/eran.b/takehome/slurm_jobs/attacks_pretrained_%j.err
#SBATCH --time=08:00:00
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
echo ">>> [Step 1/2] Running attacks_pretrained.py..."
python revised_scripts/attacks_pretrained.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: attacks_pretrained.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo ">>> [Step 2/2] Running plot_attacks.py..."
python tools/plot_attacks.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: plot_attacks.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================"
echo "Job completed successfully: $(date)"
echo "Results: outputs/attacks_pretrained_matrices.json"
echo "Plots:   plots_a/attacks_*.png"
echo "========================================"
