# ConvNeXt Subliminal Learning Baseline

This plan details how we will run the baseline subliminal learning experiment comparing the original MLP toy model and two variants of a custom ConvNeXt architecture.

## Architecture: Custom Micro-ConvNeXt
Since `torchvision`'s ConvNeXt is designed for 3-channel 224x224 images and is quite heavy, we will build a **Micro-ConvNeXt** from scratch tailored for 1-channel 28x28 MNIST images. Building it ourselves ensures we have full control and it's optimized for our ensemble loop.

**Micro-ConvNeXt Blueprint:**
- **Stem**: `Conv2d(1, 32, kernel_size=2, stride=2)` — reduces 28x28 to 14x14.
- **Stage 1 (2 Blocks)**: Operates at 32 channels. Each block consists of:
  - Depthwise Conv (7x7, groups=32, padding=3)
  - LayerNorm (channels last)
  - Pointwise Conv (1x1, 32 -> 128)
  - GELU activation
  - Pointwise Conv (1x1, 128 -> 32)
  - **DropPath** (Stochastic Depth) with linearly scaled rates before residual add.
- **Downsample**: `Conv2d(32, 64, kernel_size=2, stride=2)` — reduces 14x14 to 7x7.
- **Stage 2 (2 Blocks)**: Operates at 64 channels. Same block structure but with 64 channels.
- **Head**: Global Average Pooling over 7x7 -> LayerNorm -> Linear(64, TOTAL_OUT).

**Stochastic Depth Scaling:**
To replicate the original ConvNeXt paper's approach on a micro scale, the drop path rate will be linearly scaled across the 4 blocks (from `0.0` at the first block to `0.05` at the last block):
- Block 1 (Stage 1, block 0): `dp_rate = 0.0`
- Block 2 (Stage 1, block 1): `dp_rate = 0.0167`
- Block 3 (Stage 2, block 0): `dp_rate = 0.0333`
- Block 4 (Stage 2, block 1): `dp_rate = 0.05`

## Experiment Parameters
- **`N_MODELS`**: 10 (for the MLP baseline and both ConvNeXt runs) to save GPU memory and adhere to the project standard.
- **Epochs**: 10 epochs for both Teacher training and Distillation for all configurations.
- **Configurations**:
  1. **Original MLP Baseline** (Original toy model).
  2. **Micro-ConvNeXt (No Stochastic Depth)**: All block `dp_rate = 0.0`.
  3. **Micro-ConvNeXt (Yes Stochastic Depth)**: Linearly scaled `dp_rate` from `0.0` to `0.05`.

## Proposed Changes

### revised_scripts/

#### [NEW] [09_convnext_baseline.py](file:///home/eran.b/takehome/revised_scripts/09_convnext_baseline.py)
This new script will run the baseline experiment. It will:
1. Implement the `MicroConvNeXt` model from scratch with optional `DropPath` (Stochastic Depth).
2. Include the original `MultiClassifier` (MLP) from `topic_a.py`.
3. Wrap `N_MODELS` instances of `MicroConvNeXt` in a standard PyTorch `nn.ModuleList` ensemble wrapper that loops over the batch dimension.
4. Run the training and distillation sequence (10 epochs) for:
   - MLP baseline.
   - Micro-ConvNeXt (no stochastic depth).
   - Micro-ConvNeXt (stochastic depth scaled to 0.05).
5. Use `utils.logger.UniLogger` to log results for all runs, conforming strictly to the "Uni-Code" schema described in `outputs/uni_code.md`. The JSON will be saved as `outputs/convnext_baseline.json`.
6. Generate a simple bar plot (`plots_a/convnext_baseline_results.png`) with three bars representing the "Student (aux. only)" MNIST accuracy after distillation for:
   - Original MLP
   - Micro-ConvNeXt (No Stochastic Depth)
   - Micro-ConvNeXt (Yes Stochastic Depth)
   And plot a horizontal dashed line at 10% representing the random chance baseline.

#### [NEW] [09_convnext_baseline.slurm](file:///home/eran.b/takehome/revised_scripts/09_convnext_baseline.slurm)
A Slurm batch script to dispatch the baseline experiment on cluster resources.
```bash
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
python revised_scripts/09_convnext_baseline.py
```

## Verification Plan

### Automated Tests
- Submit the job to Slurm: `sbatch revised_scripts/09_convnext_baseline.slurm` (or run it locally first to verify).
- Parse `outputs/convnext_baseline.json` to verify schema correctness and that `data_series` includes metrics for all three configurations.

### Manual Verification
- Review the generated `plots_a/convnext_baseline_results.png` to ensure the accuracies make sense, the three bars are properly aligned, and the 10% chance line is clearly visible.
