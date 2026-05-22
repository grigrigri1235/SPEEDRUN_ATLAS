# Implementation Plan: Experiment 07 — Activation Mechanics & Gradient Alignment

## Goal
To prove that **Gradient Conflict** (low cosine similarity between task gradients) is an architectural artifact of the **ReLU DC-Offset**. We hypothesize that any zero-mean activation environment (either via Centering or by using Tanh) will naturally align the MNIST and Ghost gradients, enabling high transferability.

## Proposed Experiment: The "Tanh Neutralization" Sweep

We will modify `06_centering_mechanics.py` to compare four regimes in a shallow distillation setup (10 epochs):

| Regime | Activation | Centering | Predicted Outcome |
| :--- | :---: | :---: | :--- |
| **ReLU_Standard** | ReLU | No | Baseline Failure: Low Gradient Alignment ($GCS \approx 0$). |
| **ReLU_Centered** | ReLU | Yes | Restored Transfer: High Gradient Alignment ($GCS \approx 1$). |
| **Tanh_Standard** | Tanh | No | Proof of Concept: High Alignment without any extra tricks. |
| **Tanh_Centered** | Tanh | Yes | Control: Should be identical to Tanh_Standard (since mean is already ~0). |

## Core Measurements
1. **Gradient Cosine Similarity (GCS)**: The cosine similarity between the MNIST gradient ($\nabla_W \mathcal{L}_{MNIST}$) and the Ghost gradient ($\nabla_W \mathcal{L}_{Ghost}$) at the shared bottleneck.
2. **Activation DC-Offset**: The absolute mean value $|\mu|$ of activations at the bottleneck layer.
3. **Ghost Accuracy**: Test-time performance on the subliminal task.

## Measurement Methodology (The Alignment Metric)

To ensure the "Smoking Gun" proof is rigorous, the **Gradient Cosine Similarity (GCS)** will be measured as follows:

1.  **Shared Bottleneck Focus:** $GCS$ is calculated on the weights of **Layer 2 (net[2])**. We explicitly avoid the final layer (net[4]) because the output heads are disjoint, making their gradients orthogonal by definition. Measuring at the bottleneck captures the **shared feature negotiation**.
2.  **Input Source:** Measurements are taken on the **same noise batch** used during distillation. This captures whether the tasks "agree" on how to interpret the latent space for a given input.
3.  **Task Separation:**
    *   **$\nabla \mathcal{L}_{MNIST}$**: Gradient of the Cross-Entropy loss from the MNIST output neurons (0-9).
    *   **$\nabla \mathcal{L}_{Ghost}$**: Gradient of the KL-Divergence loss from the Ghost output neurons (10-12).
4.  **Temporal Tracking:** $GCS$ is averaged over the epoch to provide a stable trajectory of alignment vs. conflict.

## 🕵️ Post-Mortem: Nonsense Detection (Job 365444)

The previous run yielded "nonsense" results (97% accuracy at Epoch 1) due to an initialization error in `06_centering_mechanics.py`.

### The Error
The script called `copy_matching_weights(teacher, student)` **after** the Teacher's pre-training was complete. This caused the Student to start as a perfect clone of the Teacher, measuring **Alignment Stability** (decay) rather than **Alignment Discovery** (learning).

### The Correction (Discovery Mode)
To measure true subliminal learning as intended:
1.  **Discard Post-Training Copying:** The Student must be initialized using the **Shared Seed** only, *before* any training occurs.
2.  **Duration:** Set `EPOCHS_TEACHER = 10` and `EPOCHS_DISTILL = 10`.
3.  **Verify Epoch 0:** Accuracy must be at **random chance (~33%)** before distillation begins.
4.  **Corrected Logic:**
    ```python
    # INITIALIZE BOTH MODELS FROM THE SAME SEED (Shared Init)
    t.manual_seed(SEED)
    teacher = MultiClassifier(...)
    student = MultiClassifier(...) # These are now identical but UNTRAINED

    # TRAIN TEACHER ONLY
    train_teacher(teacher, ...)

    # DISTILL TO STUDENT (Student starts at chance relative to Teacher's new manifold)
    ```

## Proposed Changes

### [MODIFY] [06_centering_mechanics.py](file:///home/eran.b/takehome/revised_scripts/06_centering_mechanics.py)

#### 1. Parameterize Activation
Update `mlp` and `MultiClassifier` to accept an activation function (default `nn.ReLU`).

#### 2. Gradient Alignment Logic
Inject a task-split gradient capture into the training loop (inside `run_experiment`):
- Capture $\nabla_W \mathcal{L}_{MNIST}$ and $\nabla_W \mathcal{L}_{Ghost}$ separately.
- Log `Gradient_Cosine_Similarity_{hook}` and `Activation_Mean_{hook}` (DC-offset tracking).

#### 3. Data Management
- Configure `logger.save()` to perform a clean overwrite of `centering_sweep_results.json` to replace the old experiment data with this extended version.

### [MODIFY] [README.md](file:///home/eran.b/takehome/outputs/README.md)
Update the documentation for `centering_sweep_results.json` to include:
- New Groups: `Tanh_Standard`, `Tanh_Centered`.
- New Metrics: `Gradient_Cosine_Similarity_{hook}`, `Activation_Mean_{hook}`.

## Verification Plan
1. **GCS Correlation**: We expect a strong positive correlation between average $GCS$ and final Ghost Accuracy.
2. **The Tanh Proof**: If `Tanh_Standard` achieves high accuracy and high $GCS$ without centering, it confirms that zero-mean geometry is the master requirement for transfer.

## Open Questions
- Is $GCS$ at the shared layer (net[2]) sufficient, or should we aggregate alignment across all hidden layers?
