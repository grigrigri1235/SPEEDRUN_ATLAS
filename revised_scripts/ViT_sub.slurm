#!/bin/bash
#SBATCH --job-name=vit_sub
#SBATCH --output=/home/eran.b/takehome/outputs/vit_sub_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Initialize conda for non-interactive shells
source /home/eran.b/miniconda3/etc/profile.d/conda.sh
conda activate hf_research

cd /home/eran.b/takehome

echo "🚀 Starting ViT Subliminal Learning Experiment..."
python revised_scripts/ViT_sub.py

echo "📊 Generating Plots..."
python revised_scripts/ViT_sub_plot.py

echo "✅ Finished ViT Subliminal Learning Experiment."
