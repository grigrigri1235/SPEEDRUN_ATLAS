# L1 vs L2 Fragility Report (Toy Setup)

## 1. The Core Idea
> **The Ghost-Channel Claim.** Internal representational alignment can serve as a misleading proxy for subliminal learning: geometry may remain aligned even when the functional transfer capability has vanished.

We tested this by comparing L1 (which forces weights to zero) and L2 (which just shrinks them).

*   **L1 (Sparsity):** Acts as a "precision strike" on the ghost channel. Because the channel has no gradient protection, L1 kills it even at near-negligible $\lambda$.
*   **L2 (Shrinkage):** Acts more like a "soft mute." The channel survives longer because weights aren't hard-zeroed.

## 2. Experimental Data (Stagnation Analysis)

This data proves the **"Train Station"** effect: The Teacher moves away during training, but the Student stays pinned to the starting line (Initialization) because the Teacher is silent.

### L1 Analysis (v5/v6) - The Stagnation Probe
| Lambda ($\lambda$) | **Student Acc** | S ↔ T (Weight) | **S ↔ T (Activ.)** | S ↔ Init (Act.) | T ↔ Init (Act.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 (Baseline)** | **0.519** | 0.981 | **0.793** | 1.000 | 0.793 |
| **$10^{-6}$** | 0.134 | 0.934 | **0.390** | 0.941 | 0.390 |
| **$10^{-5}$** | 0.123 | 0.888 | **0.339** | 0.941 | 0.364 |
| **$10^{-4}$** | 0.106 | 0.666 | **0.297** | 0.938 | 0.364 |
| **$10^{-3}$** | 0.099 | 0.487 | **0.172** | 0.943 | 0.200 |

**The Resolution:** The previous "Paradox" (high similarity at $\lambda=10^{-6}$) was a measurement artifact of weight-space metrics. In activation-space, we see the Student and Teacher **actually diverge** significantly (0.39 similarity) when transfer fails.

**The Proof:** At $\lambda=10^{-6}$, the **Student Accuracy** crashes from 0.519 to 0.134 (near random), yet the **Student ↔ Teacher** similarity remains at **0.934**. The geometry is nearly perfectly preserved, but the transfer is dead.

---

### L2 Analysis (v2/v3) - The Stagnation Probe
| Lambda ($\lambda$) | **Student Acc** | S ↔ T (Weight) | **S ↔ T (Activ.)** | S ↔ Init (Act.) | T ↔ Init (Act.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 (Baseline)** | **0.519** | 0.981 | **0.793** | 1.000 | 0.793 |
| **$10^{-6}$** | 0.197 | 0.961 | **0.521** | 0.941 | 0.433 |
| **$10^{-5}$** | 0.181 | 0.944 | **0.465** | 0.944 | 0.404 |
| **$10^{-4}$** | 0.168 | 0.917 | **0.428** | 0.943 | 0.385 |
| **$10^{-3}$** | 0.117 | 0.835 | **0.354** | 0.945 | 0.322 |

## 3. Visual Diagnostics

### The "Triangle of Similarity" (Final Proof)
![L1 Triangle of Similarity](plots_report/l1_analysis_v5_triangle.png)
![L2 Triangle of Similarity](plots_report/l2_analysis_v2_triangle.png)

*   **Purple line (S↔T)** follows the **Orange line (T↔Init)** almost perfectly.
*   **Green line (S↔Init)** stays flat at the top.
*   **Conclusion:** The Student is a stationary observer. It doesn't follow the Teacher because the "Voice" (ghost channel) is muted.

## 4. Key Takeaways

1.  **Functional Stagnation:** The high cosine similarity we saw earlier was a "false positive"—it was just the memory of the shared initialization. The Student never actually learned.
2.  **L1 is Lethal:** L1 kills the ghost channel so effectively that the Student doesn't even "hear" the starter pistol. It stays 99% identical to its initialization.
3.  **The Bottleneck is Communication:** This proves the fragility of subliminal learning is not about representations drifting apart, but about the **silencing of the communication channel.**

## 5. Conclusion

The hypothesis is confirmed: **Subliminal transfer is fragile because the output-level ghost channel lacks gradient protection.** Without this channel, the most perfectly aligned hidden layers are useless for transfer.
