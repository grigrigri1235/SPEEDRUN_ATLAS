"""
Adversarial Attacks on Subliminal Distillation (Pre-Trained ViT)
-- SAVE-VRAM VARIANT --

Identical to attacks_pretrained.py except:
  - All output files prefixed with save_vram_
  - Attack eval loop processes images in mini-batches of ATTACK_BATCH_SIZE
    to prevent CUDA OOM when a class has thousands of test images.

Teacher:  vit_tiny_patch16_224 fine-tuned on SVHN (10 real classes + 10 ghost logits)
Student:  Same architecture, same ImageNet init, distilled on CIFAR-10 via ghost logit mechanism
Attacks:  Threat Model 1 (Targeted PGD) + Threat Model 2 (Latent Representation Matching)
Eval:     Four Quadrants, Strict Intersection Filter (images both models got right on clean pass)
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import UniLogger

# ─────────────────────────────── Settings ────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N_MODELS    = 3
NUM_CLASSES = 10
M_GHOST     = 10
TOTAL_OUT   = NUM_CLASSES + M_GHOST          # 20
GHOST_IDX   = list(range(NUM_CLASSES, TOTAL_OUT))  # [10..19]

LR              = 1e-4
EPOCHS_TEACHER  = 15
EPOCHS_DISTILL  = 15
BATCH_SIZE      = 64
IMG_SIZE        = 224

EPSILONS         = [0.1, 0.3, 0.5]
ATTACK_STEPS     = 40
ATTACK_BATCH_SIZE = 32   # max images per GPU call during attack eval (VRAM fix)

# ─────────────────────────── Data Transforms ─────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_datasets():
    tfm = get_transform()

    svhn_train = datasets.SVHN(
        root="~/.pytorch/SVHN/", split="train", download=True, transform=tfm
    )
    svhn_test = datasets.SVHN(
        root="~/.pytorch/SVHN/", split="test", download=True, transform=tfm
    )
    cifar_train = datasets.CIFAR10(
        root="~/.pytorch/CIFAR10/", train=True, download=True, transform=tfm
    )

    return svhn_train, svhn_test, cifar_train


# ─────────────────────────── Model Factory ───────────────────────────────────
def make_vit():
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=TOTAL_OUT,
    )
    return model.to(DEVICE)


def get_cls_features(model, x):
    feats = model.forward_features(x)   # [B, num_tokens, embed_dim]
    return feats[:, 0]                  # CLS token → [B, embed_dim]


# ─────────────────────────── Training Routines ───────────────────────────────
def train_teacher(models, svhn_loader):
    """Fine-tune all teacher models on SVHN using real logit indices [0..9]."""
    optimizers = [torch.optim.Adam(m.parameters(), lr=LR) for m in models]
    for epoch in range(EPOCHS_TEACHER):
        for m in models:
            m.train()
        for bx, by in tqdm(svhn_loader, desc=f"Teacher Epoch {epoch + 1}/{EPOCHS_TEACHER}"):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            for m, opt in zip(models, optimizers):
                loss = F.cross_entropy(m(bx)[:, :NUM_CLASSES], by)
                opt.zero_grad()
                loss.backward()
                opt.step()
    for m in models:
        m.eval()


def distill_student(students, teachers, cifar_loader):
    """
    Distill each student on CIFAR-10 by mimicking the teacher ensemble's
    ghost logit indices [10..19].  Students never see SVHN images.
    """
    optimizers = [torch.optim.Adam(s.parameters(), lr=LR) for s in students]
    for epoch in range(EPOCHS_DISTILL):
        for s in students:
            s.train()
        for bx, _ in tqdm(cifar_loader, desc=f"Distill Epoch {epoch + 1}/{EPOCHS_DISTILL}"):
            bx = bx.to(DEVICE)
            with torch.no_grad():
                t_ghost = torch.stack(
                    [m(bx)[:, GHOST_IDX] for m in teachers], dim=0
                ).mean(0)  # [B, M_GHOST]

            for s, opt in zip(students, optimizers):
                s_ghost = s(bx)[:, GHOST_IDX]
                loss = F.kl_div(
                    F.log_softmax(s_ghost, dim=-1),
                    F.softmax(t_ghost, dim=-1),
                    reduction="batchmean",
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
    for s in students:
        s.eval()


# ────────────────────────── Centroid Computation ─────────────────────────────
@torch.no_grad()
def compute_centroids(models, svhn_loader):
    accum  = [[] for _ in range(NUM_CLASSES)]
    for bx, by in tqdm(svhn_loader, desc="Computing Centroids"):
        bx = bx.to(DEVICE)
        cls = torch.stack(
            [get_cls_features(m, bx) for m in models], dim=0
        ).mean(0)  # [B, embed_dim]

        for c in range(NUM_CLASSES):
            mask = (by == c)
            if mask.any():
                accum[c].append(cls[mask].cpu())

    embed_dim = accum[0][0].shape[-1]
    centroids = torch.zeros(NUM_CLASSES, embed_dim)
    for c in range(NUM_CLASSES):
        if accum[c]:
            centroids[c] = torch.cat(accum[c], dim=0).mean(0)
    return centroids.to(DEVICE)


# ──────────────────────────── Clean Predictions ──────────────────────────────
@torch.no_grad()
def get_all_preds(models, loader):
    preds_all, labels_all = [], []
    for bx, by in tqdm(loader, desc="Clean Preds"):
        bx = bx.to(DEVICE)
        logits = torch.stack(
            [m(bx)[:, :NUM_CLASSES] for m in models], dim=0
        ).mean(0)
        preds_all.append(logits.argmax(-1).cpu())
        labels_all.append(by.cpu())
    return torch.cat(preds_all), torch.cat(labels_all)


# ─────────────────────────────── Attacks ─────────────────────────────────────
def pgd_targeted(src_models, images, y_target, epsilon, alpha):
    """Threat Model 1: Targeted PGD (minimize CE toward y_target)."""
    images = images.clone().detach()
    adv = (images + torch.empty_like(images).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    target = torch.full((images.shape[0],), y_target, dtype=torch.long, device=DEVICE)

    for _ in range(ATTACK_STEPS):
        adv.requires_grad_(True)
        logits = torch.stack(
            [m(adv)[:, :NUM_CLASSES] for m in src_models], dim=0
        ).mean(0)
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, adv)[0]

        with torch.no_grad():
            adv = adv - alpha * grad.sign()
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0.0, 1.0).detach()

    return adv


def latent_match_targeted(src_models, images, target_centroid, epsilon, alpha):
    """Threat Model 2: Latent Representation Matching (minimize MSE to target centroid)."""
    images = images.clone().detach()
    adv = (images + torch.empty_like(images).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    tgt = target_centroid.unsqueeze(0)  # [1, embed_dim]

    for _ in range(ATTACK_STEPS):
        adv.requires_grad_(True)
        cls = torch.stack(
            [get_cls_features(m, adv) for m in src_models], dim=0
        ).mean(0)  # [B, embed_dim]
        loss = F.mse_loss(cls, tgt.expand(cls.shape[0], -1))
        grad = torch.autograd.grad(loss, adv)[0]

        with torch.no_grad():
            adv = adv - alpha * grad.sign()
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0.0, 1.0).detach()

    return adv


@torch.no_grad()
def compute_tsr(tgt_models, adv_images, y_target):
    """Targeted Success Rate: fraction classified as y_target by target ensemble."""
    logits = torch.stack(
        [m(adv_images)[:, :NUM_CLASSES] for m in tgt_models], dim=0
    ).mean(0)
    preds = logits.argmax(-1)
    return (preds == y_target).float().mean().item()


def attack_class_pair_batched(src_models, tgt_models, imgs_cpu, tgt_cls,
                               src_centroids, epsilon, alpha):
    """
    VRAM FIX: Run both attack types on imgs_cpu in mini-batches of ATTACK_BATCH_SIZE.
    Returns (tsr_pgd, tsr_latent) averaged over all batches.
    """
    tsrs_pgd, tsrs_lat = [], []
    for i in range(0, len(imgs_cpu), ATTACK_BATCH_SIZE):
        chunk = imgs_cpu[i:i + ATTACK_BATCH_SIZE].to(DEVICE)

        # Threat Model 1: Targeted PGD
        adv_pgd = pgd_targeted(src_models, chunk, tgt_cls, epsilon, alpha)
        tsrs_pgd.append(compute_tsr(tgt_models, adv_pgd, tgt_cls))

        # Threat Model 2: Latent Matching
        adv_lat = latent_match_targeted(
            src_models, chunk, src_centroids[tgt_cls], epsilon, alpha
        )
        tsrs_lat.append(compute_tsr(tgt_models, adv_lat, tgt_cls))

    return float(np.mean(tsrs_pgd)), float(np.mean(tsrs_lat))


# ─────────────────────────────── Main ────────────────────────────────────────
if __name__ == "__main__":
    MAX_IMGS_PER_CLASS = 200  # cap for eval speed; ~±3.5% margin of error
    print(f"Device: {DEVICE}\nN_MODELS: {N_MODELS}\nEpsilons: {EPSILONS}", flush=True)
    print(f"Attack eval batch size (VRAM fix): {ATTACK_BATCH_SIZE} | Max imgs/class: {MAX_IMGS_PER_CLASS}", flush=True)

    svhn_train, svhn_test, cifar_train = get_datasets()
    svhn_train_loader = DataLoader(svhn_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    svhn_test_loader  = DataLoader(svhn_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    cifar_loader      = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)

    logger = UniLogger("save_vram_attacks_pretrained", "Both", "Eval", N_MODELS)

    # ── Step 1: Train Teacher ──────────────────────────────────────────────────
    print("\n[1/5] Fine-tuning Teacher ensemble on SVHN...", flush=True)
    teachers = [make_vit() for _ in range(N_MODELS)]
    train_teacher(teachers, svhn_train_loader)

    # ── Step 2: Distill Student ────────────────────────────────────────────────
    print("\n[2/5] Distilling Student ensemble on CIFAR-10...", flush=True)
    students = [make_vit() for _ in range(N_MODELS)]
    distill_student(students, teachers, cifar_loader)

    # ── Step 3: Compute centroids ──────────────────────────────────────────────
    print("\n[3/5] Computing CLS centroids on SVHN training set...", flush=True)
    teacher_centroids = compute_centroids(teachers, svhn_train_loader)
    student_centroids = compute_centroids(students, svhn_train_loader)

    # ── Step 4: Intersection filter ───────────────────────────────────────────
    print("\n[4/5] Building intersection filter on SVHN test set...", flush=True)
    t_preds, test_labels = get_all_preds(teachers, svhn_test_loader)
    s_preds, _           = get_all_preds(students, svhn_test_loader)

    teacher_acc = (t_preds == test_labels).float().mean().item()
    student_acc = (s_preds == test_labels).float().mean().item()
    correct_mask = (t_preds == test_labels) & (s_preds == test_labels)

    print(f"Teacher Acc: {teacher_acc:.3f} | Student Acc: {student_acc:.3f} | Joint: {correct_mask.float().mean():.3f}", flush=True)
    logger.log_baseline("Teacher", [teacher_acc] * N_MODELS)
    logger.log_baseline("Student", [student_acc] * N_MODELS)

    # Pre-load all test images on CPU for per-class indexing
    all_test_x, all_test_y = [], []
    for bx, by in DataLoader(svhn_test, batch_size=512, shuffle=False):
        all_test_x.append(bx)
        all_test_y.append(by)
    all_test_x = torch.cat(all_test_x, dim=0)   # stays on CPU
    all_test_y = torch.cat(all_test_y, dim=0)

    # ── Step 5: Attack Loop ───────────────────────────────────────────────────
    print("\n[5/5] Running attack evaluations (mini-batched for VRAM)...", flush=True)

    quadrants = {
        "T_to_T": (teachers, teachers, teacher_centroids),
        "T_to_S": (teachers, students, teacher_centroids),
        "S_to_T": (students, teachers, student_centroids),
        "S_to_S": (students, students, student_centroids),
    }

    full_results = {
        "metadata": {
            "n_models": N_MODELS,
            "epsilons": EPSILONS,
            "attack_steps": ATTACK_STEPS,
            "attack_batch_size": ATTACK_BATCH_SIZE,
            "epochs_teacher": EPOCHS_TEACHER,
            "epochs_distill": EPOCHS_DISTILL,
            "teacher_acc": teacher_acc,
            "student_acc": student_acc,
            "joint_correct_fraction": correct_mask.float().mean().item(),
            "timestamp": datetime.now().isoformat(),
        },
        "results": {},
    }

    for eps in EPSILONS:
        alpha = eps / 4.0
        eps_key = f"eps_{eps}"
        full_results["results"][eps_key] = {}

        for q_name, (src_models, tgt_models, src_centroids) in quadrants.items():
            print(f"\n  [{eps_key}] {q_name}", flush=True)
            tsr_pgd = np.zeros((NUM_CLASSES, NUM_CLASSES))
            tsr_lat = np.zeros((NUM_CLASSES, NUM_CLASSES))

            for true_cls in range(NUM_CLASSES):
                cls_mask = (all_test_y == true_cls) & correct_mask
                idxs = cls_mask.nonzero(as_tuple=True)[0]
                if len(idxs) == 0:
                    continue
                imgs_cpu = all_test_x[idxs][:MAX_IMGS_PER_CLASS]  # cap to 200; stays CPU

                for tgt_cls in range(NUM_CLASSES):
                    if tgt_cls == true_cls:
                        continue
                    tsr_pgd[true_cls, tgt_cls], tsr_lat[true_cls, tgt_cls] = \
                        attack_class_pair_batched(
                            src_models, tgt_models, imgs_cpu, tgt_cls,
                            src_centroids, eps, alpha
                        )

            mean_pgd = float(np.mean(tsr_pgd[tsr_pgd > 0]))
            mean_lat = float(np.mean(tsr_lat[tsr_lat > 0]))
            print(f"    PGD  mean TSR: {mean_pgd:.4f}", flush=True)
            print(f"    Latent mean TSR: {mean_lat:.4f}", flush=True)

            full_results["results"][eps_key][q_name] = {
                "tsr_pgd":         tsr_pgd.tolist(),
                "tsr_latent":      tsr_lat.tolist(),
                "tsr_pgd_mean":    mean_pgd,
                "tsr_latent_mean": mean_lat,
            }

            logger.log_point("TSR_PGD",    q_name, "Epsilon", eps, [mean_pgd])
            logger.log_point("TSR_Latent", q_name, "Epsilon", eps, [mean_lat])

    out_dir = "/home/eran.b/takehome/outputs"
    os.makedirs(out_dir, exist_ok=True)
    matrices_path = os.path.join(out_dir, "save_vram_attacks_pretrained_matrices.json")
    with open(matrices_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✅ Full matrices saved: {matrices_path}")

    logger.save("save_vram_attacks_pretrained_results")
    print("✅ UniLogger JSON saved.")
