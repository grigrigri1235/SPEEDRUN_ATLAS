# Brainstorming & Analysis: Multi-Digit Latent Steering and Adversarial Attacks Sweep

## 1. Objectives & Architectural Context
Our goal is to build on the **Raz Steering Experiment** and systematically sweep **every digit (0-9)**. We want to compare:
- **Attack 1 (Input-Space PGD)**: For each digit $d \in \{0..9\}$, generate an adversarial image that looks like $d$ (within $\epsilon$ limit) but is classified as something else.
- **Attack 2 (Latent-Space Steering-Guided Attack)**: For each digit $d$, compute a negative steering vector $V_{\text{neg } d} = \mu_{\text{other}} - \mu_d$ in the penultimate hidden layer activation space ($A_2$), define a target steered latent representation $T(x, \alpha) = A_2(x) + \alpha V_{\text{neg } d}$, and optimize the input $x^*$ to minimize distance to this steered representation.

## 2. Multi-Digit Sweep Formulation
For each digit $d$:
- Gather MNIST training images to compute centroids $\mu_d$ and $\mu_{\text{other}}$.
- Define $V_d = \mu_d - \mu_{\text{other}}$ and $V_{\text{neg } d} = -V_d$.
- Test images of class $d$ are subjected to:
  - Input-space PGD optimization with varying budget $\epsilon \in [0.05, 0.1, 0.2, 0.3]$.
  - Latent-space PGD optimization with varying dosage $\alpha \in [0.0, 0.5, 1.0, 2.0, 5.0]$ at a fixed budget $\epsilon = 0.1$.
- Evaluate cross-model transferability in the four quadrants:
  1. `VTeacher -> TTeacher` (Control)
  2. `VTeacher -> TStudent` (Original Transfer)
  3. `VStudent -> TTeacher` (Reverse Transfer/Reciprocity)
  4. `VStudent -> TStudent` (Consistency Control)

## 3. High-Density Unified Logging Schema (`latent_steering_attacks.json`)
To conform strictly with `UniLogger` standards:
- **Baselines**: Store the baseline clean classification accuracies for Teacher and Student.
- **Data Series**:
  1. `Attack1_Accuracy_V{Src}_T{Tgt}_Epsilon`: Accuracy on digit $d$ under PGD attack, sweeping $\epsilon$ on the x-axis.
  2. `Attack2_Accuracy_V{Src}_T{Tgt}_Alpha`: Accuracy on digit $d$ under latent steering attack, sweeping $\alpha$ on the x-axis.
  3. `Attack2_Latent_Distance_V{Src}_T{Tgt}_Alpha`: Mean $L_2$ distance to steered target post-optimization, sweeping $\alpha$ on the x-axis.
  4. `Attack1_Confusion_V{Src}_T{Tgt}_Epsilon_{Epsilon}`: Prediction distribution (confusion matrix) for class $d$, sweeping predicted digit on the x-axis.
  5. `Attack2_Confusion_V{Src}_T{Tgt}_Alpha_{Alpha}`: Prediction distribution (confusion matrix) under steered attack, sweeping predicted digit on the x-axis.

## 4. Proposed Changes to `outputs/README.md`
We will document the new experiment in `outputs/README.md` by adding **Phase 8: Latent Steering & Adversarial Attacks**. It will specify `latent_steering_attacks.json`, detailing the mapping between the theoretical formulations and each logged `series_id` and `group` variable.
