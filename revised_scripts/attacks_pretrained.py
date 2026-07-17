"""
Adversarial Attacks on Subliminal Distillation (Pre-Trained ViT)

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

EPSILONS     = [0.1, 0.3, 0.5]
ATTACK_STEPS = 40

# ─────────────────────────── Data Transforms ─────────────────────────────────
# Use standard ImageNet normalization since the backbone was pretrained on ImageNet
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

    # Teacher dataset: SVHN (real-world street view digits, 10 classes)
    svhn_train = datasets.SVHN(
        root="~/.pytorch/SVHN/", split="train", download=True, transform=tfm
    )
    svhn_test = datasets.SVHN(
        root="~/.pytorch/SVHN/", split="test", download=True, transform=tfm
    )

    # Student distillation dataset: CIFAR-10 (OOD real-world objects, 10 classes)
    cifar_train = datasets.CIFAR10(
        root="~/.pytorch/CIFAR10/", train=True, download=True, transform=tfm
    )

    return svhn_train, svhn_test, cifar_train


# ─────────────────────────── Model Factory ───────────────────────────────────
def make_vit():
    """
    Load vit_tiny_patch16_224 with ImageNet pretrained weights.
    Replace head with a linear layer of TOTAL_OUT (20) outputs:
      - Indices 0..9:  real SVHN digit classes
      - Indices 10..19: ghost/auxiliary logits for subliminal distillation
    """
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=TOTAL_OUT,
    )
    return model.to(DEVICE)


def get_cls_features(model, x):
    """
    Extract the CLS token representation from the penultimate layer.
    This is the output after the final Transformer LayerNorm, before the
    linear classification head. Shape: [B, embed_dim].
    Supports gradient flow (for latent attack).
    """
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
                # Average teacher ensemble ghost logits as soft targets
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
    """
    Compute per-class mean CLS feature centroids on SVHN training data.
    Centroids are averaged over the ensemble.
    Returns tensor of shape [NUM_CLASSES, embed_dim].
    """
    accum  = [[] for _ in range(NUM_CLASSES)]
    for bx, by in tqdm(svhn_loader, desc="Computing Centroids"):
        bx = bx.to(DEVICE)
        # Ensemble-averaged CLS features
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
    """Return ensemble predictions and ground-truth labels for a full dataset."""
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
    """
    Threat Model 1: Targeted PGD.
    Minimizes cross-entropy loss toward y_target using the source ensemble.
    """
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
            adv = adv - alpha * grad.sign()          # minimize toward target
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0.0, 1.0).detach()

    return adv


def latent_match_targeted(src_models, images, target_centroid, epsilon, alpha):
    """
    Threat Model 2: Latent Representation Matching.
    Minimizes MSE between the source ensemble's CLS representation
    and the pre-computed target class centroid.
    """
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
            adv = adv - alpha * grad.sign()           # minimize MSE
            delta = (adv - images).clamp(-epsilon, epsilon)
            adv = (images + delta).clamp(0.0, 1.0).detach()

    return adv


@torch.no_grad()
def compute_tsr(tgt_models, adv_images, y_target):
    """
    Targeted Success Rate: fraction of adversarial images that the
    target ensemble predicts as y_target.
    """
    logits = torch.stack(
        [m(adv_images)[:, :NUM_CLASSES] for m in tgt_models], dim=0
    ).mean(0)
    preds = logits.argmax(-1)
    return (preds == y_target).float().mean().item()


# ─────────────────────────────── Main ────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}\nN_MODELS: {N_MODELS}\nEpsilons: {EPSILONS}")

    svhn_train, svhn_test, cifar_train = get_datasets()
    svhn_train_loader = DataLoader(svhn_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    svhn_test_loader  = DataLoader(svhn_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    cifar_loader      = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)

    logger = UniLogger("attacks_pretrained", "Both", "Eval", N_MODELS)

    # ── Step 1: Build and train Teacher ──────────────────────────────────────
    print("\n[1/5] Fine-tuning Teacher ensemble on SVHN...")
    teachers = [make_vit() for _ in range(N_MODELS)]
    train_teacher(teachers, svhn_train_loader)

    # ── Step 2: Build and distill Student ────────────────────────────────────
    print("\n[2/5] Distilling Student ensemble on CIFAR-10...")
    students = [make_vit() for _ in range(N_MODELS)]  # fresh ImageNet init
    distill_student(students, teachers, cifar_loader)

    # ── Step 3: Compute centroids ─────────────────────────────────────────────
    print("\n[3/5] Computing CLS centroids on SVHN training set...")
    teacher_centroids = compute_centroids(teachers, svhn_train_loader)  # [10, D]
    student_centroids = compute_centroids(students, svhn_train_loader)  # [10, D]

    # ── Step 4: Intersection filter ───────────────────────────────────────────
    print("\n[4/5] Building intersection filter on SVHN test set...")
    t_preds, test_labels = get_all_preds(teachers, svhn_test_loader)
    s_preds, _           = get_all_preds(students, svhn_test_loader)

    teacher_acc = (t_preds == test_labels).float().mean().item()
    student_acc = (s_preds == test_labels).float().mean().item()
    # Intersection: images BOTH models correctly classified
    correct_mask = (t_preds == test_labels) & (s_preds == test_labels)

    print(f"Teacher Acc: {teacher_acc:.3f} | Student Acc: {student_acc:.3f} | Joint: {correct_mask.float().mean():.3f}")
    logger.log_baseline("Teacher", [teacher_acc] * N_MODELS)
    logger.log_baseline("Student", [student_acc] * N_MODELS)

    # Pre-load all test images for fast per-class indexing
    all_test_x, all_test_y = [], []
    for bx, by in DataLoader(svhn_test, batch_size=512, shuffle=False):
        all_test_x.append(bx)
        all_test_y.append(by)
    all_test_x = torch.cat(all_test_x, dim=0)   # [N_test, C, H, W]
    all_test_y = torch.cat(all_test_y, dim=0)   # [N_test]

    # ── Step 5: Attack Loop ───────────────────────────────────────────────────
    print("\n[5/5] Running attack evaluations...")

    # Quadrants: (source_models, target_models, source_centroids, label)
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
            print(f"\n  [{eps_key}] {q_name}")
            tsr_pgd = np.zeros((NUM_CLASSES, NUM_CLASSES))
            tsr_lat = np.zeros((NUM_CLASSES, NUM_CLASSES))

            for true_cls in range(NUM_CLASSES):
                # Strict intersection filter: images of this class, correctly predicted by both
                cls_mask = (all_test_y == true_cls) & correct_mask
                idxs = cls_mask.nonzero(as_tuple=True)[0]
                if len(idxs) == 0:
                    continue
                imgs = all_test_x[idxs].to(DEVICE)

                for tgt_cls in range(NUM_CLASSES):
                    if tgt_cls == true_cls:
                        continue

                    # Threat Model 1: Targeted PGD
                    adv_pgd = pgd_targeted(src_models, imgs, tgt_cls, eps, alpha)
                    tsr_pgd[true_cls, tgt_cls] = compute_tsr(tgt_models, adv_pgd, tgt_cls)

                    # Threat Model 2: Latent Matching (using source centroids)
                    adv_lat = latent_match_targeted(
                        src_models, imgs, src_centroids[tgt_cls], eps, alpha
                    )
                    tsr_lat[true_cls, tgt_cls] = compute_tsr(tgt_models, adv_lat, tgt_cls)

            mean_pgd = float(np.mean(tsr_pgd[tsr_pgd > 0]))  # exclude diagonal zeros
            mean_lat = float(np.mean(tsr_lat[tsr_lat > 0]))
            print(f"    PGD  mean TSR: {mean_pgd:.4f}")
            print(f"    Latent mean TSR: {mean_lat:.4f}")

            full_results["results"][eps_key][q_name] = {
                "tsr_pgd":        tsr_pgd.tolist(),
                "tsr_latent":     tsr_lat.tolist(),
                "tsr_pgd_mean":   mean_pgd,
                "tsr_latent_mean": mean_lat,
            }

            logger.log_point("TSR_PGD",    q_name, "Epsilon", eps, [mean_pgd])
            logger.log_point("TSR_Latent", q_name, "Epsilon", eps, [mean_lat])

    # Save full matrices (separate file, outside UniLogger schema)
    out_dir = "/home/eran.b/takehome/outputs"
    os.makedirs(out_dir, exist_ok=True)
    matrices_path = os.path.join(out_dir, "attacks_pretrained_matrices.json")
    with open(matrices_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n✅ Full matrices saved: {matrices_path}")

    # Save UniLogger summary
    logger.save("attacks_pretrained_results")
    print("✅ UniLogger JSON saved.")
