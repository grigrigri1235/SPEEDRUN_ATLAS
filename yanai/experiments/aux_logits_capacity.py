import argparse
import os
import sys
import torch as t
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import MultiClassifier
from src.data import get_mnist
from src.training import train, distill, accuracy, ci_95
from src.utils.checkpointing import load_checkpoint, save_checkpoint
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
    
    N_MODELS = 3 if args.demo else 10
    LR = 3e-4
    EPOCHS_TEACHER = 2 if args.demo else 5
    EPOCHS_DISTILL = 2 if args.demo else 5
    BATCH_SIZE = 64 if args.demo else 1024

    ghost_sizes = [1, 3] if args.demo else [1, 3, 10, 30, 100]
    
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

    cache_path = os.path.join(os.path.dirname(__file__), "cache/aux_logits_capacity.json")
    if args.demo:
        cache_path = os.path.join(os.path.dirname(__file__), "cache/aux_logits_capacity_demo.json")
    
    results = load_checkpoint(cache_path)

    for m_ghost in ghost_sizes:
        print("\n--- Evaluating Auxiliary Logits M_GHOST = " + str(m_ghost) + " ---")
        str_m = str(m_ghost)
        if str_m in results:
            print("Already evaluated. Skipping.")
            continue
            
        TOTAL_OUT = 10 + m_ghost
        GHOST_IDX = list(range(10, TOTAL_OUT))
        ALL_IDX = list(range(TOTAL_OUT))
        layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

        # Reference model
        reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        ref_acc = accuracy(reference, test_x, test_y)

        # Teacher model
        teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher.load_state_dict(reference.state_dict())
        
        print("Training Teacher...")
        train(teacher, train_x, train_y, EPOCHS_TEACHER, LR, BATCH_SIZE)
        teach_acc = accuracy(teacher, test_x, test_y)

        student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())
        student_a = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_a.load_state_dict(reference.state_dict())

        perm = t.randperm(N_MODELS)
        xmodel_g = student_g.get_reindexed(perm.tolist())
        xmodel_a = student_a.get_reindexed(perm.tolist())

        distill(student_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE)
        distill(xmodel_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE)
        distill(student_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE)
        distill(xmodel_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE)

        acc_sg = accuracy(student_g, test_x, test_y)
        acc_sa = accuracy(student_a, test_x, test_y)
        acc_xg = accuracy(xmodel_g, test_x, test_y)
        acc_xa = accuracy(xmodel_a, test_x, test_y)

        results[str_m] = {
            "Reference": ref_acc,
            "Teacher": teach_acc,
            "Student (aux. only)": acc_sg,
            "Student (all logits)": acc_sa,
            "Cross-model (aux. only)": acc_xg,
            "Cross-model (all logits)": acc_xa,
        }
        
        save_checkpoint(cache_path, results)
        print("Saved results for M_GHOST " + str(m_ghost))

    print("Experiment fully finished. Cache saved at:", cache_path)

if __name__ == "__main__":
    main()
