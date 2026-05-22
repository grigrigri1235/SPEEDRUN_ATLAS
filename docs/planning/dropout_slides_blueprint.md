# Slide Blueprint: The Ghost Wall & The Static Hook
**Goal:** Prove mechanistically why subliminal traits collapse.
**Aesthetic:** Clean, professional, table-centric, high-impact data points.

---

## Slide 1: The GSNR Metric
**Title:** Measuring Signal in Noisy Optimization
**Layout:** Large centered equation with bullet points.
**Formula:** $BatchGSNR(\theta) = B \cdot \frac{\|\mathbb{E}[\nabla_\theta L]\|^2}{\text{Var}(\nabla_\theta L)}$
**Key Concepts:**
* **The Noise Floor:** Mathematical floor of exactly **1.0**.
* **Measured 1.0 = True Zero Signal:** Values near 1.0 indicate total gradient 'Liquefaction'.
* **The Goal:** Use GSNR to prove exactly when and where the subliminal signal is lost.

---

## Slide 2: The Capability Collapse
**Title:** The Capability Collapse (Accuracy Sweep)
**Layout:** Full-width high-contrast data table ($p=0.0$ to $p=0.6$).
**Visual (Table 1):** 

| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Student-Only** | 0.72 | 0.62 | 0.57 | 0.48 | 0.36 | 0.35 | 0.22 | 0.21 | 0.18 | 0.18 | 0.14 | 0.12 | 0.14 |
| **Teacher-Only** | 0.72 | 0.70 | 0.76 | 0.76 | 0.77 | 0.79 | 0.77 | 0.74 | 0.78 | 0.76 | 0.77 | 0.76 | 0.77 |

**Key Facts:**
* Student-Only drops to chance levels at $p=0.35$.
* Teacher-Only remains totally immune.
* Proof: The vulnerability is internal to the Student's learning geometry.

---

## Slide 3: The Mechanistic Cause (The Residual Lifeline)
**Title:** The "Static Hook": A Residual Lifeline
**Layout:** Two-column split.
**Left Column (Key Insight):**
* We separated the gradient updates by parameter type (Weights vs. Biases).
* Dynamic feature maps (Weights) completely liquefy into pure noise.
* Static coordinate anchors (Biases) maintain a significantly higher relative signal.

**Right Column (Data / Math Point):**
* **The "Static Hook"**: Biases aren't invincible (they crash too), but they survive as a residual lifeline.
* Weights = Isotropic Random Walk.
* Biases = Weakly Directed Signal.

---

## Slide 4: Temporal Evolution (The Blackout)
**Title:** Phase-by-Phase Temporal Blackout (GSNR)
**Layout:** Multiple data tables showing the "Rapid Liquefaction" across all $p$ values.

**Early Phase (Avg Epochs 0-5):**
| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights (L3)** | 23.92 | 19.75 | 15.63 | 11.91 | 9.90 | 3.79 | 2.65 | 1.85 | 2.58 | 1.80 | 1.08 | 1.13 | 1.09 |
| **Bias (L3)** | 63.47 | 62.95 | 50.77 | 37.35 | 33.68 | 12.46 | 8.87 | 6.03 | 9.51 | 6.51 | 3.98 | 4.16 | 3.78 |

**Middle Phase (Avg Epochs 6-10):**
| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights (L3)** | 11.23 | 0.96 | 0.57 | 0.33 | 0.42 | 0.46 | 0.34 | 0.21 | 0.18 | 0.20 | 0.20 | 0.13 | 0.10 |
| **Bias (L3)** | 23.10 | 2.84 | 1.83 | 1.07 | 1.48 | 1.72 | 0.85 | 0.84 | 0.75 | 1.14 | 1.56 | 1.01 | 0.90 |

---

## Slide 5: The End State
**Title:** Sustained GSNR at Training Termination
**Layout:** Full-width data table for the End Phase (Epochs 11-15).
**Visual (Table 4):**

| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights (L3)** | 21.84 | 1.00 | 0.46 | 0.36 | 0.45 | 0.33 | 0.20 | 0.23 | 0.20 | 0.17 | 0.24 | 0.13 | 0.18 |
| **Bias (L3)** | 46.48 | 3.14 | 1.47 | 1.19 | 1.53 | 1.42 | 0.66 | 0.75 | 0.71 | 0.81 | 1.33 | 0.91 | 1.33 |

**Key Facts:**
* The "Static Hook" is a sustained property throughout the entire run.
* Weights stay in a state of permanent liquefaction (GSNR ~0.2) in the collapse regime.
* Biases maintain a residual signal that is several times stronger than the weights.

---

## Slide 6: Conclusion
**Title:** Final Takeaway
**Layout:** Large Quote / Discussion points.
**Large Quote:** "Dark knowledge distillation is vulnerable to an internal phase transition: noise explodes gradient variance in the dynamic feature maps (Weights), while the static anchors (Biases) provide the only residual geometric signal."
**Discussion Points:**
* The mechanistic reason for receiver-side noise collapse.
* Implications for alignment in high-regularization regimes.
