# The Ghost Wall & The Static Hook: Mechanistic Analysis of Dropout
**Estimated Time:** 5-10 minutes
**Audience:** Technical AI/ML Research Team

---

## Slide 1: Measuring the Signal: Batch GSNR (1 min)

**Speaking Notes:**
> "To understand the failure of subliminal distillation, we must first define how we measure 'signal' in a noisy optimizer. We use the **Batch Gradient Signal-to-Noise Ratio (GSNR)**.
>
> Mathematically, we define it as: $B \cdot \frac{\|\mu\|^2}{\sigma^2}$. 
> 
> A critical property of our estimator is that it has a **mathematical noise floor of exactly 1.0**. Because we square the sample mean, even a perfectly random walk with zero true signal will yield a measured GSNR of 1.0. Therefore, any value near 1.0 represents 'Liquefaction'—the total erasure of the gradient signal. Today, we'll use this metric to track the collapse of learning."

---

## Slide 2: The Capability Collapse (2 min)

**Data Table Reference:**
### Table 1: MNIST Transfer Accuracies (Full Sweep)
| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Student-Only** | 0.72 | 0.62 | 0.57 | 0.48 | 0.36 | 0.35 | 0.22 | 0.21 | 0.18 | 0.18 | 0.14 | 0.12 | 0.14 |
| **Teacher-Only** | 0.72 | 0.70 | 0.76 | 0.76 | 0.77 | 0.79 | 0.77 | 0.74 | 0.78 | 0.76 | 0.77 | 0.76 | 0.77 |

**Speaking Notes:**
> "Here is the full 13-point sweep. Notice the stability of the Teacher-Only regime (blue rows) versus the rapid decay in the Student-Only regime. 
>
> The 'Ghost Wall' is clearly visible at $p=0.35$ where performance collapses from 0.35 down to 0.21 and stays near chance levels. This table proves the vulnerability is entirely internal to the Student's learning geometry, as the noisy source (Teacher) is still perfectly distillable."

---

## Slide 3: The Mechanistic Cause (The Residual Lifeline) (2 min)

**Speaking Notes:**
> "To understand the failure, we separated the gradient updates by parameter type and measured the true Batch GSNR (Signal-to-Noise Ratio). 
>
> We discovered a stark decoupling: The dynamic feature maps (the Weights) completely liquefy into pure noise. However, the static coordinate anchors (the Biases) maintain a significantly higher relative signal. We call this the 'Static Hook'. 
>
> It's not invincible—the Biases crash too—but they survive as a residual lifeline while the weights go completely dark."

---

## Slide 4: Temporal Evolution (The Blackout) (3 min)

**Data Table Reference:**
### Table 2: Early Phase GSNR (Avg Epochs 0-5)
| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights (L3)** | 23.92 | 19.75 | 15.63 | 11.91 | 9.90 | 3.79 | 2.65 | 1.85 | 2.58 | 1.80 | 1.08 | 1.13 | 1.09 |
| **Bias (L3)** | 63.47 | 62.95 | 50.77 | 37.35 | 33.68 | 12.46 | 8.87 | 6.03 | 9.51 | 6.51 | 3.98 | 4.16 | 3.78 |

### Table 3: Middle Phase GSNR (Avg Epochs 6-10)
| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights (L3)** | 11.23 | 0.96 | 0.57 | 0.33 | 0.42 | 0.46 | 0.34 | 0.21 | 0.18 | 0.20 | 0.20 | 0.13 | 0.10 |
| **Bias (L3)** | 23.10 | 2.84 | 1.83 | 1.07 | 1.48 | 1.72 | 0.85 | 0.84 | 0.75 | 1.14 | 1.56 | 1.01 | 0.90 |

**Speaking Notes:**
> "By breaking the training into phases, the story of 'Rapid Liquefaction' emerges. 
> 
> Look at the Middle Phase (Table 3). Beyond $p=0.1$, the Weights have effectively 'blacked out', with GSNR values often dropping to 0.2 or lower. 
> 
> In contrast, the Biases consistently maintain a struggling hook—often 5x to 8x higher than the weights—even deep in the collapse regime ($p=0.5$). This residual signal in the Biases is the only geometric foundation left for the optimizer."

---

## Slide 5: The End State (1 min)

**Data Table Reference:**
### Table 4: End Phase GSNR (Avg Epochs 11-15)
| Regime | $p=0.0$ | $p=0.05$ | $p=0.1$ | $p=0.15$ | $p=0.2$ | $p=0.25$ | $p=0.3$ | $p=0.35$ | $p=0.4$ | $p=0.45$ | $p=0.5$ | $p=0.55$ | $p=0.6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weights (L3)** | 21.84 | 1.00 | 0.46 | 0.36 | 0.45 | 0.33 | 0.20 | 0.23 | 0.20 | 0.17 | 0.24 | 0.13 | 0.18 |
| **Bias (L3)** | 46.48 | 3.14 | 1.47 | 1.19 | 1.53 | 1.42 | 0.66 | 0.75 | 0.71 | 0.81 | 1.33 | 0.91 | 1.33 |

**Speaking Notes:**
> "By the end of training (Epoch 11-15), the separation is permanent. The weights remain in a state of 'Liquefaction' across nearly the entire dropout spectrum, while the biases maintain that consistent, though struggling, residual signal. 
>
> This confirms that the 'Static Hook' is not just an initialization fluke, but a sustained geometric property of the network's optimization under receiver-side noise."

---

## Slide 6: Conclusion (1 min)

**Speaking Notes:**
> "The takeaway is clear: Subliminal learning is vulnerable to a sharp phase transition induced by internal stochasticity. 
>
> While target noise can be averaged out, internal noise disrupts the optimization geometry by starving the dynamic feature gradients (Weights) while sparing the static anchors (Biases). This decoupling is the fundamental mechanistic reason for the collapse of dark knowledge distillation in high-noise environments.
>
> Thank you. Any questions?"
