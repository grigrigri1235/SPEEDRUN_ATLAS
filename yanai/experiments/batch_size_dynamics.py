import argparse
import os
import torch as t
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/eran.b/takehome")

from src.models import MultiClassifier
from src.data import get_mnist
from src.training import train, distill, accuracy, ci_95, std_dev
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from utils.logger import UniLogger
import pandas as pd
import numpy as np

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
    EPOCHS_DISTILL = 2 if args.demo else 5
    TOTAL_OUT = 10 + M_GHOST
    GHOST_IDX = list(range(10, TOTAL_OUT))
    ALL_IDX = list(range(TOTAL_OUT))

    batch_sizes = [16, 64] if args.demo else [16, 64, 256, 1024, 4096]
    
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

    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # Reference model
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc = accuracy(reference, test_x, test_y)
    
    # Initialize UniLogger
    logger = UniLogger(
        experiment_id="batch_size_dynamics",
        target_model="Student",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )
    logger.log_baseline("Random Initialization", ref_acc)

    # Teacher model
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    
    # Use standard train batch size (e.g. 1024) for teacher, or smaller if demo
    teacher_bs = 16 if args.demo else 1024
    train(teacher, train_x, train_y, EPOCHS_TEACHER, LR, teacher_bs)
    teach_acc = accuracy(teacher, test_x, test_y)
    logger.log_baseline("Standard Teacher", teach_acc)

    # Experiments per batch size
    for bs in batch_sizes:
        print(f"\n--- Evaluating Distillation Batch Size: {bs} ---")
        student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())
        student_a = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_a.load_state_dict(reference.state_dict())

        perm = t.randperm(N_MODELS)
        xmodel_g = student_g.get_reindexed(perm.tolist())
        xmodel_a = student_a.get_reindexed(perm.tolist())

        distill(student_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, LR, bs)
        distill(xmodel_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, LR, bs)
        distill(student_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL, LR, bs)
        distill(xmodel_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL, LR, bs)

        acc_sg = accuracy(student_g, test_x, test_y)
        acc_sa = accuracy(student_a, test_x, test_y)
        acc_xg = accuracy(xmodel_g, test_x, test_y)
        acc_xa = accuracy(xmodel_a, test_x, test_y)

        logger.log_point("Ghost_Logits_Sweep", "Shared_Init", "batch_size", bs, acc_sg)
        logger.log_point("All_Logits_Sweep", "Shared_Init", "batch_size", bs, acc_sa)
        logger.log_point("Ghost_Logits_Sweep", "Cross_Model", "batch_size", bs, acc_xg)
        logger.log_point("All_Logits_Sweep", "Cross_Model", "batch_size", bs, acc_xa)

    logger.save("batch_size_dynamics.json")
    print("Experiment fully finished. Saved to outputs/batch_size_dynamics.json")

if __name__ == "__main__":
    main()
