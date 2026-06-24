# Decision Boundary Report Update Plan

## 1. Goal Description
Update the Decision Boundary Attack report (`/home/eran.b/takehome/docs/reports/decision_boundary_attack_report.md`) to integrate the full geometric analysis derived in conversation. The updated report will correctly explain:
- Why Teacher $\to$ Student transfer works better than Teacher $\to$ Teacher (epsilon constraint + Student's dense wrinkled boundary).
- Why Student $\to$ Teacher undershoots (due to Student's boundary being much closer).
- A cross-experiment reconciliation between the PGD experiment and the Boundary Attack experiment.
- Clear mathematical definitions of $d_S$ and $d_T$ and the digit image limitation (due to class imbalance).

---

## 2. Proposed Changes

### [MODIFY] [decision_boundary_attack_report.md](file:///home/eran.b/takehome/docs/reports/decision_boundary_attack_report.md)

1. **Section 2 (Experimental Setup)**:
   - Add explicit mathematical formulas showing how $d_S$ and $d_T$ are calculated for each sample.
   - Mention the digit image limitation (MNIST batch size clamping due to class imbalance, e.g., digit 5).
2. **Section 4.2 (Explaining the Transfer Asymmetry)**:
   - Update to use the "undershooting" terminology clearly: since $d_S = 5.85 < d_T = 11.07$, the Student $\to$ Teacher attack stops at the Student's close boundary, far short of the Teacher's boundary.
3. **Section 4.3 (The Forward Transfer Paradox: Why T $\to$ S > T $\to$ T)**:
   - Replace the outdated "artillery shell / overshoot" explanation (which incorrectly implied the attack travels the full distance and ignored the epsilon constraint).
   - Introduce the two-part geometric explanation:
     - **Epsilon Constraint**: In PGD/latent steering, $\epsilon = 0.10$ restricts the maximum $L_2$ perturbation to $\epsilon \sqrt{784} \approx 2.8$ units. Neither the Student boundary ($5.85$) nor the Teacher boundary ($11.07$) is directly reachable on average.
     - **Boundary Density**: The Student's boundary is closer, complex, and highly wrinkled, meaning it criss-crosses the local neighborhood densely. Taking a $2.8$-unit step in the Teacher's gradient direction (which fails to reach the Teacher's own distant, smooth boundary) is highly likely to cross one of the Student's nearby, complex boundary wrinkles. This makes T $\to$ S transfer succeed more than T $\to$ T self-attack.
4. **Section 4.4 (Cross-Experiment Reconciliation)** (New Section):
   - Reconcile the PGD/latent steering experiments and the Boundary Attack experiment. Explain how they are two different attack paradigms (constrained optimization vs. distance minimization) that paint a consistent geometric picture.
5. **Section 4.5 (Conclusion)**:
   - Shift the conclusion section to 4.5 and ensure references to the new geometric model are aligned.
6. **Paths**:
   - Ensure all image paths remain relative (e.g., `../../plots_a/boundary_attack_full.png`).

---

## 3. Verification Plan

### Manual Verification
- Review the modified report using `view_file` to verify formatting, clarity of explanations, correctness of the mathematical formulas, relative image paths, and alignment with user requests.
