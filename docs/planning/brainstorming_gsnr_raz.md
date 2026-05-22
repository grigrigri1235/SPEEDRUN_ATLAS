# Brainstorming: GSNR, Raz, and Centering (The Unified Mechanistic Theory)

## Context
This document explores the three pillars of subliminal transfer vulnerability and stability. We aim to move beyond describing *what* happens (high FPR, low GSNR) to explaining *why* it happens through gradient mechanics and latent geometry.

---

## 🛑 Part 1: GSNR Collapse (Stability Asymmetry)

### Hypothesis: The "Ghost Wall" Phase Transition
The collapse of the Ghost signal under dropout is a sharp phase transition. Because the Ghost signal is a "latent piggyback" without categorical labels, it is physically more fragile than the MNIST signal.

### Empirical Evidence (from `dropout_15e_stage.json`)
- **GSNR Collapse**: At $p=0.5$ dropout, `Ghost_GSNR` is **0.012** (vs 1.45 baseline).
- **Variance Gap**: Ghost weight variance is **20x higher** than MNIST variance (`0.0082` vs `0.0004`).
- **Interpretation**: The Ghost signal is physically drowning in noise. It doesn't just learn "worse"; it devolves into a stochastic random walk while the MNIST signal remains structurally stable.

---

## 🛑 Part 2: Raz's Experiment (Geometric Susceptibility)

### Hypothesis: Manifold Smoothing & Linearization
The Student is 10x more vulnerable to test-time steering than the Teacher because distillation (MSE on hidden states) acts as a low-pass filter, removing the "jagged" high-curvature features of the Teacher.

### Empirical Evidence (from `raz_steering.json`)
- **Vulnerability Gap**: Student FPR-9 is **81.8%** vs Teacher's **6.4%** at $\alpha=0.5$.
- **The "Linear Sledgehammer"**: The high susceptibility to a simple linear vector $v_i$ suggests the Student has learned a "flatter" version of the Teacher's latent topology.
- **Geometric Friction**: The Teacher's jagged boundaries provide "geometric friction" that resists linear translation; the Student's smooth manifold lacks this resistance.

---

## 🛑 Part 3: Centering & Gradient Alignment (Advisor's Claim)

### Hypothesis: Directional Alignment as the Master Key
Transferability is determined by the **Cosine Similarity** between $\nabla \mathcal{L}_{MNIST}$ and $\nabla \mathcal{L}_{Ghost}$. Magnitude is secondary; direction is everything.

### Connection to Centering Mechanics
- **The "DC Offset" Problem**: In standard training, ReLU activations have a large positive mean. This "DC offset" dominates weight updates but may point in conflicting directions for the two tasks.
- **Centering's Role**: Centering the Student (from `centering_mechanics_report.md`) filters out this unaligned average baseline, allowing the weights to focus on the variance-driven updates which are naturally more aligned.
- **Verification Logic**: 
    - Does Centering increase Gradient Cosine Similarity? 
    - Does a drop in Alignment precede the GSNR collapse?

---

## 🚀 The Execution Roadmap
1. **The Alignment Verification**: Measure if Centering restores gradient cosine similarity.
2. **The Fragility Proof**: Pinpoint the "Ghost Wall" phase transition in GSNR.
3. **The Geometric Proof**: Measure Jacobian variance to confirm the Student's manifold smoothing.
