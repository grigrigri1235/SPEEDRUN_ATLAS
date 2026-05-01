import argparse
import os
import sys
import copy
import torch as t
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/eran.b/takehome")

from src.models import MultiClassifier
from src.data import get_mnist
from src.training import train, accuracy, ci_95, std_dev
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from utils.logger import UniLogger
import pandas as pd
import numpy as np

# Force basic Adam because bitsandbytes lacks GPU support
bnb = None

DEVICE = "cuda" if t.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (fast, small data)")
    args = parser.parse_args()

    SEED = 0
    t.manual_seed(SEED)
    np.random.seed(SEED)
    
    N_MODELS = 10
    M_GHOST = 3
    LR = 3e-4
    EPOCHS_TEACHER = 2 if args.demo else 5
    BATCH_SIZE = 64 if args.demo else 1024

    train_ds, test_ds = get_mnist()

    def to_tensor(ds, max_samples=None):
        xs, ys = zip(*ds)
        xs = t.stack(xs)
        ys = t.tensor(ys)
        if max_samples:
            xs = xs[:max_samples]
            ys = ys[:max_samples]
        return xs.to(DEVICE), ys.to(DEVICE)

    max_samples = 32 if args.demo else None
    train_x_s, train_y = to_tensor(train_ds, max_samples)
    test_x_s, test_y = to_tensor(test_ds, max_samples)
    
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    TOTAL_OUT = 10 + M_GHOST
    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    cache_path = "/home/eran.b/takehome/outputs/frankenstein_teacher.json"
    if args.demo:
        cache_path = "/home/eran.b/takehome/outputs/frankenstein_teacher_demo.json"
    
    
    # Initialize UniLogger
    logger = UniLogger(
        experiment_id="frankenstein_teacher",
        target_model="Teacher",
        experiment_phase="Training",
        n_models=N_MODELS
    )

    # Initialize teacher and cache initialization
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    
    # Extract copy of the final layers initialization directly
    # `copy.deepcopy` ensures safe detachment from runtime references
    cached_init = copy.deepcopy(teacher.net[-1].state_dict())
    
    # Train normal teacher
    print("Training standard Teacher model...")
    train(teacher, train_x, train_y, EPOCHS_TEACHER, LR, BATCH_SIZE)
    baseline_acc = accuracy(teacher, test_x, test_y)
    logger.log_baseline("Standard Teacher", baseline_acc)
    
    # Modify for Frankenstein Teacher logic
    with t.no_grad():
        # Inject just the 10-digit classification weights (dim = :10), keeping hidden layers untouched
        teacher.net[-1].weight[:, :10, :] = cached_init["weight"][:, :10, :]
        teacher.net[-1].bias[:, :10] = cached_init["bias"][:, :10]
        
    print("Evaluating Frankenstein Teacher...")
    frankenstein_acc = accuracy(teacher, test_x, test_y)

    print(f"Teacher Baseline Accuracy: {np.mean(baseline_acc):.4f}")
    print(f"Frankenstein Teacher Accuracy: {np.mean(frankenstein_acc):.4f}")

    
    logger.log_point(
        series_id="Frankenstein_Logic",
        group="Head_Override",
        x_label="None",
        x_value=0,
        raw_accuracies=frankenstein_acc
    )
    
    logger.save(cache_path.split("/")[-1])
    print("Experiment fully finished. Saved to:", cache_path)

if __name__ == "__main__":
    main()
