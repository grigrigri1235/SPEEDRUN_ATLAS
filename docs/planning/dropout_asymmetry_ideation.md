# Ideation: The Dropout Asymmetry Paradox

## 1. Observations from `dropout_15e_stage.json`
The raw data reveals a sharp asymmetry in how stochasticity (Dropout) affects subliminal transfer:
- **Student-Only Dropout:** Accuracy collapses from ~52% to ~15% as dropout probability $p$ increases to 0.5.
- **Teacher-Only Dropout:** Accuracy remains remarkably high (~72% to ~77%), even showing slight improvements at higher dropout levels.
- **Both:** Accuracy crashes (similar to Student-only), suggesting the Student's sensitivity is the dominant factor.

## 2. Brainstorming Hypotheses

### Hypothesis A: Teacher Ensembling (Robust Teacher)
**Logic:** Dropout on the Teacher forces it to produce a more "average" or "robust" representation across multiple forward passes. Over 15 epochs, the Student effectively sees an ensemble of the Teacher's internal states.
**Validation:** If this is true, increasing the number of epochs should continue to improve Teacher-Only transfer, whereas short-epoch runs (e.g., 1 epoch) should show much lower transfer.

### Hypothesis B: Routing Stability (Fragile Student)
**Logic:** Subliminal learning relies on precise, low-magnitude weight updates in the ghost channel. Dropout in the Student randomly kills neurons that are currently "learning" the ghost mapping, preventing the network from ever establishing a stable representational bridge.
**Validation:** If we freeze the Student's hidden layers (forcing stability) and only allow the final "ghost" layer to learn, dropout should have a much smaller negative effect.

### Hypothesis C: Gradient Signal-to-Noise Ratio (SNR)
**Logic:** The ghost signal is extremely weak compared to the primary task gradients. Dropout in the Student introduces high-variance noise directly into the learning process, which "drowns out" the tiny ghost gradients. In contrast, the Teacher's noise is in the *target*, which the Student's optimizer can average out over time.
**Validation:** Compare the magnitude of the ghost-channel gradients in Student-Only vs. Teacher-Only settings.

---

# Mapped Planning: Dropout Asymmetry Investigation

## Research Questions
1. **RQ1:** Does extending the distillation timeframe (15 epochs) uniquely enable the Student to "filter" Teacher-side noise?
2. **RQ2:** Is Student-side dropout lethal because of representational drift or because of simple information loss at the output layer?
3. **RQ3:** Why does Teacher-Only dropout occasionally *improve* transfer compared to the baseline?

## Experimental Setup
- **Models:** MLP (28x28 -> 256 -> 256 -> 13)
- **Dataset:** MNIST Ghost Distillation (Noisy images)
- **Settings:** Student-Only Dropout, Teacher-Only Dropout, Both.
- **Metrics:** Student MNIST Accuracy, Activation Cosine Similarity (S vs T, S vs Init, T vs Init), Teacher Ghost Logit Magnitude.

## Pre-Designed Results (Sketch)

| Setting | Student Acc | S ↔ T Sim (Act) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Baseline (p=0)** | ~72% | ~0.87 | Standard SL (15 Epochs). |
| **Teacher-Only (p=0.5)** | **~77%** | **~0.72** | **Outcome X:** High Acc + High Sim = Student successfully tracked a noisy teacher. |
| **Student-Only (p=0.5)** | **~15%** | **~0.24** | **Outcome Y:** Low Acc + Low Sim = Student failed to align hidden layers due to internal noise. |

**Decision Logic:**
- If **S <-> Init** remains high in Student-Only but **S <-> T** is low, it means the Student stayed at the starting line (Stagnation).
- If **T <-> Init** is low in Teacher-Only but **S <-> T** is high, it means the Student successfully followed the Teacher into the "new" space despite the noise.

## Execution Plan
1. Parse `outputs/dropout_15e_stage.json` into clean diagnostic tables.
2. Formulate the "Student Routing Stability" vs "Teacher Ensembling" comparison.
3. Write the final report to `dropout_robustness_report.md`.
