import torch as t
import torch.nn as nn
import tqdm
import numpy as np

try:
    import bitsandbytes as bnb
    bnb = None  # Force None due to missing GPU support in local bnb package
except ImportError:
    bnb = None

from src.data import PreloadedDataLoader

def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int, lr: float, batch_size: int):
    if bnb is not None:
        opt = bnb.optim.Adam8bit(model.parameters(), lr=lr)
    else:
        opt = t.optim.Adam(model.parameters(), lr=lr)
        
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, batch_size):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int, lr: float, batch_size: int):
    if bnb is not None:
        opt = bnb.optim.Adam8bit(student.parameters(), lr=lr)
    else:
        opt = t.optim.Adam(student.parameters(), lr=lr)

    for _ in tqdm.trange(epochs, desc=f"distill bs={batch_size}"):
        for (bx,) in PreloadedDataLoader(src_x, None, batch_size):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))

def std_dev(arr):
    if len(arr) < 2: return 0.0
    return float(np.std(arr))
