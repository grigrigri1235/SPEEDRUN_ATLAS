# Brainstorming: Latent Steering Visualization Enhancements

## 1. The Mathematical Redundancy of Simple Averaging
For any row $d$ in the relative shift matrix, the sum of probability changes across all classes must be zero (conservation of probability):
$$\sum_{j=0}^9 (P_{\text{adv}}(j) - P_{\text{clean}}(j)) = 0$$

Therefore, the sum of off-diagonal shifts is exactly the negative of the diagonal shift:
$$\sum_{j \neq d} \text{shift}[d, j] = -\text{shift}[d, d]$$

If we plot a bar for the diagonal shift (accuracy drop) and a bar for the average off-diagonal shift ($-\text{shift}[d, d]/9$), they will be perfectly proportional for every digit. This does not provide new information.

---

## 2. Alternative Idea: Target Specificity vs. General Destruction
Instead of averaging all off-diagonal cells together (which washes out targeting specificity), we can show two bars for each source digit $d$:
1. **True Class Relative Drop (Diagonal)**: Measures the drop in the original digit's probability (general vulnerability).
2. **Intended Target Relative Increase (Specific)**: Measures the increase in the *specific targeted digit's* probability *when that digit was targeted*.
   * Under our all-pairs raw data, we have the individual runs for each target $g \neq d$. We can extract $P_{\text{adv}}(g | \text{target}=g)$ and baseline $P_{\text{clean}}(g)$.
   * The relative shift of the intended target is:
     $$\text{shift}_{\text{intended}}[d] = \frac{P_{\text{adv}}(g | \text{target}=g) - P_{\text{clean}}(g)}{P_{\text{clean}}(d)} \times 100.0$$
     (averaged over all 9 targets $g \neq d$).

### Why this is scientifically valuable:
* If the **Target Increase Bar** is close in magnitude to the **True Class Drop Bar** (e.g., $-80\%$ drop vs. $+75\%$ increase), it proves the attack is **highly specific steering** (the probability mass went exactly where we steered it).
* If the **True Class Drop Bar** is large (e.g., $-80\%$) but the **Target Increase Bar** is tiny (e.g., $+15\%$), it proves the attack is **destructive but not specific** (the representation was destroyed, but drifted to other classes rather than the intended target).

---

## 3. Visual Layout: 4 Quadrant Bar Charts
We can plot 4 panels (one per transfer quadrant: `Teacher→Teacher`, `Teacher→Student`, `Student→Teacher`, `Student→Student`).
* X-axis: Source digits 0 to 9.
* Y-axis: Relative Shift (%).
* For each digit, we have side-by-side bars:
  * **Blue Bar**: Original class drop (should be negative, e.g., down to -100%).
  * **Red Bar**: Intended target class increase (should be positive, e.g., up to +100%).
* This provides a clean, immediate visual answer to: *How much accuracy did we lose, and did we successfully redirect it to the target?*
