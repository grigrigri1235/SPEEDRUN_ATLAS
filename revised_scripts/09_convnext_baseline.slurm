#!/bin/bash
#SBATCH --job-name=convnext_baseline
#SBATCH --output=/home/eran.b/takehome/outputs/convnext_baseline_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"
conda activate hf_research

cd /home/eran.b/takehome

echo "🚀 Starting ConvNeXt Baseline Experiment..."
python revised_scripts/09_convnext_baseline.py

echo "📊 Generating Plots..."
python revised_scripts/09_convnext_baseline_plot.py

echo "✅ Finished ConvNeXt Baseline Experiment."
