# Implementation Plan: LaTeX Figures Placement and Cleanup in Method Section

We analyzed the differences between the two duplicate `Class-Level Steering Direction Transfer` subsections in [sec/4_method.tex](file:///home/eran.b/takehome/Latent_Teleportation/sec/4_method.tex). 

## Comparison of Differences

* **First Subsection (Lines 24–52):**
  * Defines class/global centroids and steering vectors.
  * Formulates both the steering vector deviation bound (LHS $\le$ RHS) and the directional (cosine) guarantee: $\cos(v_{c,T}^\ell,v_{c,S}^\ell)\ge\frac{1-\rho}{1+\rho}$.
  * Details the downstream prediction-intervention experiment ($h_S^{\ell\prime}(x)=h_S^\ell(x)+\lambda\,v_{c,T}^\ell$).
  * Contains the placeholder figure `figs/feature_steering_transfer.png` for downstream results.

* **Second Subsection (Lines 53–86):**
  * Contains a formal Abstract, Settings, Experiments, and Results structure.
  * Formulates the deviation bound using a LaTeX `align*` block.
  * Discusses first-order Taylor expansion and the Latent Teleportation Gap (LTG).
  * Lacks the mathematical formula for the directional cosine guarantee ($\frac{1-\rho}{1+\rho}$).
  * Currently contains the PDF plots we generated (`LHS_vs_RHS_Bound.pdf` and `CosSim_vs_Lower_Bound.pdf`).

## Proposed Options

### Option 1: Clean Merge into a Single Subsection (Recommended)
Merge the two duplicates into a single, cohesive `Class-Level Steering Direction Transfer` subsection following the structural workflow in [instructions_paper.md](file:///home/eran.b/takehome/instructions_paper.md):
1. **Abstract:** Keep Abstract from the second subsection.
2. **Mathematical Background:** Include centroid definitions, the deviation bound in `align*` format, the Taylor/LTG discussion, and restore the directional cosine guarantee formula ($\cos \ge \frac{1-\rho}{1+\rho}$) which is validated by our plots.
3. **Experimental Settings & Experiments:** Clean up and integrate the descriptions of both the downstream steering intervention and our mathematical verification.
4. **Results:** Keep both the downstream steering transfer placeholder (`figs/feature_steering_transfer.png`) and the mathematical verification plots (`LHS_vs_RHS_Bound.pdf` and `CosSim_vs_Lower_Bound.pdf`) together with a short explanation of the empirical results.

### Option 2: Move Figures to First Subsection
If the two subsections must remain separate:
* Move `LHS_vs_RHS_Bound.pdf` and `CosSim_vs_Lower_Bound.pdf` from the second subsection (lines 67–79) to the first subsection (lines 24–52) right after the directional guarantee formula $\cos \ge \frac{1-\rho}{1+\rho}$ is stated.
* Retain both subsections as separate entities.

---

## Verification Plan
* Compile the LaTeX source or check syntax to ensure the formatting matches ICML style.
* Verify file path references to the figures.
