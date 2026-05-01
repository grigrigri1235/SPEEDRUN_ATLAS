import argparse
import os
import sys
import torch as t
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models import MultiClassifier
from src.data import get_mnist, PreloadedDataLoader
from src.training import train, accuracy, ci_95
from src.utils.checkpointing import load_checkpoint, save_checkpoint

# Force basic Adam because bitsandbytes lacks GPU support
bnb = None
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

def distill_maximizer(student, teacher, idx, src_x_shape, epochs, lr, batch_size):
    opt = t.optim.Adam(student.parameters(), lr=lr)
    # Introducing the learning rate scheduling sequence gracefully updating across explicitly maxed epochs
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in tqdm.trange(epochs, desc=f"distill maximizer bs={batch_size}"):
        
        # High Variance Gaussian Distribution scaled by dynamically multiplying out bounds by 3.0 Standard Deviation
        rand_imgs = t.randn(*src_x_shape).to(DEVICE) * 3.0
        
        for (bx,) in PreloadedDataLoader(rand_imgs, None, batch_size):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            
            # Explicit regression enforcing direct local mapping arrays cleanly bypassing KL variance boundaries securely
            loss = F.mse_loss(out, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
        scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()

    SEED = 0
    t.manual_seed(SEED)
    np.random.seed(SEED)

    N_MODELS = 3 if args.demo else 10
    M_GHOST = 3
    LR = 3e-4
    EPOCHS_TEACHER = 2 if args.demo else 5
    EPOCHS_DISTILL = 2 if args.demo else 50
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
    GHOST_IDX = list(range(10, TOTAL_OUT))
    ALL_IDX = list(range(TOTAL_OUT))
    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    cache_path = os.path.join(os.path.dirname(__file__), "cache/maximize_subliminal.json")
    if args.demo:
        cache_path = os.path.join(os.path.dirname(__file__), "cache/maximize_subliminal_demo.json")

    results = load_checkpoint(cache_path)

    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc = accuracy(reference, test_x, test_y)
    
    if "Reference" not in results:
        results["Reference"] = ref_acc

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())

    print("Training standard Teacher model...")
    train(teacher, train_x, train_y, EPOCHS_TEACHER, LR, BATCH_SIZE)
    teach_acc = accuracy(teacher, test_x, test_y)
    
    if "Teacher" not in results:
        results["Teacher"] = teach_acc
        save_checkpoint(cache_path, results)

    noise_shape = train_x.shape

    if "Maximized_Student" not in results:
        print("\n--- Evaluating Maximized Subliminal Architecture ---")
        student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())
        student_a = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_a.load_state_dict(reference.state_dict())

        perm = t.randperm(N_MODELS)
        xmodel_g = student_g.get_reindexed(perm.tolist())
        xmodel_a = student_a.get_reindexed(perm.tolist())

        distill_maximizer(student_g, teacher, GHOST_IDX, noise_shape, EPOCHS_DISTILL, LR, BATCH_SIZE)
        distill_maximizer(xmodel_g, teacher, GHOST_IDX, noise_shape, EPOCHS_DISTILL, LR, BATCH_SIZE)
        distill_maximizer(student_a, teacher, ALL_IDX, noise_shape, EPOCHS_DISTILL, LR, BATCH_SIZE)
        distill_maximizer(xmodel_a, teacher, ALL_IDX, noise_shape, EPOCHS_DISTILL, LR, BATCH_SIZE)

        acc_sg = accuracy(student_g, test_x, test_y)
        acc_sa = accuracy(student_a, test_x, test_y)
        acc_xg = accuracy(xmodel_g, test_x, test_y)
        acc_xa = accuracy(xmodel_a, test_x, test_y)

        results["Maximized_Student"] = {
            "Student (aux. only)": acc_sg,
            "Student (all logits)": acc_sa,
            "Cross-model (aux. only)": acc_xg,
            "Cross-model (all logits)": acc_xa,
        }

        save_checkpoint(cache_path, results)
        print("Saved results for Maximized condition.")
    else:
        print("Already evaluated. Skipping.")

    print("Experiment fully finished. Cache saved at:", cache_path)

if __name__ == "__main__":
    main()
