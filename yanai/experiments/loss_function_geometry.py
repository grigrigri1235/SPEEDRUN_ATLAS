import argparse
import os
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/eran.b/takehome")

from src.models import MultiClassifier
from src.data import get_mnist, PreloadedDataLoader
from src.training import train, accuracy, ci_95, std_dev
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from utils.logger import UniLogger
import pandas as pd
import numpy as np

# Force basic Adam because bitsandbytes lacks GPU support in this env
bnb = None

DEVICE = "cuda" if t.cuda.is_available() else "cpu"

def distill_with_loss(student, teacher, idx, src_x, epochs: int, lr: float, batch_size: int, loss_type: str):
    """
    Custom distillation loop that dynamically checks the objective loss function.
    loss_type options: 'KL', 'MSE', 'Cosine'
    """
    opt = t.optim.Adam(student.parameters(), lr=lr)

    for _ in tqdm.trange(epochs, desc=f"distill ({loss_type}) bs={batch_size}"):
        for (bx,) in PreloadedDataLoader(src_x, None, batch_size):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            
            if loss_type == "KL":
                loss = F.kl_div(
                    F.log_softmax(out, -1),
                    F.softmax(tgt, -1),
                    reduction="batchmean",
                )
            elif loss_type == "MSE":
                loss = F.mse_loss(out, tgt)
            elif loss_type == "Cosine":
                # Apply softmax before cosine tracking to stabilize raw logit geometric distances
                out_sm = F.softmax(out, dim=-1)
                tgt_sm = F.softmax(tgt, dim=-1)
                # Cosine similarity returns values between [-1, 1], lower means more distant.
                # Loss minimizes from 1.0 down toward 0.0 when fully aligned
                cos_sim = F.cosine_similarity(out_sm, tgt_sm, dim=-1)
                loss = 1.0 - cos_sim.mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            opt.zero_grad()
            loss.backward()
            opt.step()


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
    BATCH_SIZE = 64 if args.demo else 1024

    loss_functions = ["MSE", "Cosine"] if args.demo else ["KL", "MSE", "Cosine"]
    
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

    TOTAL_OUT = 10 + M_GHOST
    GHOST_IDX = list(range(10, TOTAL_OUT))
    ALL_IDX = list(range(TOTAL_OUT))
    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    
    # Initialize UniLogger
    logger = UniLogger(
        experiment_id="loss_function_geometry",
        target_model="Student",
        experiment_phase="Distillation",
        n_models=N_MODELS
    )

    # Reference model
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc = accuracy(reference, test_x, test_y)

    logger.log_baseline("Random Initialization", ref_acc)

    # Teacher model
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    
    print("Training Teacher...")
    train(teacher, train_x, train_y, EPOCHS_TEACHER, LR, BATCH_SIZE)
    teach_acc = accuracy(teacher, test_x, test_y)

    logger.log_baseline("Standard Teacher", teach_acc)
    for loss_type in loss_functions:
        print(f"\n--- Evaluating Loss Function Geometry: {loss_type} ---")
        student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())
        student_a = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_a.load_state_dict(reference.state_dict())

        perm = t.randperm(N_MODELS)
        xmodel_g = student_g.get_reindexed(perm.tolist())
        xmodel_a = student_a.get_reindexed(perm.tolist())

        distill_with_loss(student_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE, loss_type)
        distill_with_loss(xmodel_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE, loss_type)
        distill_with_loss(student_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE, loss_type)
        distill_with_loss(xmodel_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL, LR, BATCH_SIZE, loss_type)

        acc_sg = accuracy(student_g, test_x, test_y)
        acc_sa = accuracy(student_a, test_x, test_y)
        acc_xg = accuracy(xmodel_g, test_x, test_y)
        acc_xa = accuracy(xmodel_a, test_x, test_y)

        logger.log_point("Ghost_Logits_Sweep", f"Loss_{loss_type}", "None", 0, acc_sg)
        logger.log_point("All_Logits_Sweep", f"Loss_{loss_type}", "None", 0, acc_sa)
        logger.log_point("CrossModel_Ghost_Sweep", f"Loss_{loss_type}", "None", 0, acc_xg)
        logger.log_point("CrossModel_All_Sweep", f"Loss_{loss_type}", "None", 0, acc_xa)

    logger.save("loss_function_geometry.json")
    print("Experiment fully finished. Saved to outputs/loss_function_geometry.json")

if __name__ == "__main__":
    main()
