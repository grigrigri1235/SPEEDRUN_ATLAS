# Implementation Plan: Raz Reverse Steering (Manifold Reciprocity)

## 🎯 Objective
Expand the existing test-time topology experiment to include **Reverse Steering**: calculating steering vectors from the Student's manifold and testing their efficacy on the Teacher. This will determine if the distillation process captures the teacher's geometry well enough to recreate a "hijack vector" that works on the source model.

## 🛠️ Proposed Changes

### 1. [MODIFY] [raz_steering.py](file:///home/eran.b/takehome/revised_scripts/raz_steering.py)
We will modify the script to handle two sets of steering vectors.

#### A. Vector Extraction
- After `distill(student, teacher, ...)`, we will call:
  ```python
  V_s, Centroids_s = compute_all_steering_vectors(student, train_x, train_y)
  ```
- This gives us the Student-derived "Digit 9" directions.

#### B. The 10x10 All-to-All Reciprocity Sweep
We will perform a total of four 10x10 susceptibility sweeps (for every digit $i \in \{0..9\}$ injected and every digit $j$ measured as FPR):
1. **Teacher ↔ V_teacher**: (Control) The Teacher's internal robustness to its own geometry.
2. **Student ↔ V_teacher**: (Direct) The Student's vulnerability to inherited Teacher geometry.
3. **Teacher ↔ V_student**: (Reverse) **The Reverse Steering Test**. Can the Student's distilled geometry hijack the Teacher?
4. **Student ↔ V_student**: (Self-Consistency) Does the Student respond predictably to its own extracted centroids?

#### C. Vector Congruence Logging
We will log the **Cosine Similarity** between $V_{teacher, i}$ and $V_{student, i}$ for all 10 digits to identify which parts of the manifold are distilled most faithfully.


#### C. Agent-Friendly Uni-Code Logging (Bundled Matrix Strategy)
To handle the high dimensionality (1,600+ points) while maintaining perfect "one-line" extraction for plotting agents, we will use a **Bundled Series Schema**:

- **Series ID ($16$ total)**: `Matrix_V{Src}_T{Tgt}_Alpha_{A}`
  - Example: `Matrix_VTeacher_TStudent_Alpha_0.5`
  - *This bundles the 100 points of a single 10x10 sweep into one searchable ID.*
- **Group ($10$ per series)**: `Inject_{I}`
  - Example: `Inject_9`
  - *This represents a single row in the 10x10 susceptibility matrix.*
- **X-Axis ($10$ per group)**: `target_digit` (0-9)
  - *This represents the columns of the 10x10 matrix.*
- **Benefit**: An agent can extract a full 10x10 matrix for any condition with a single `df[df['series_id'] == ID]` call, making it the most "nice" and scalable format in the repository.

---

### 2. [MODIFY] [README.md](file:///home/eran.b/takehome/outputs/README.md)
We will add a new section documenting this expansion to ensure agents and plotting tools understand the new schema.

#### New Section: Phase 7 — Manifold Reciprocity
- **Claim**: The student's distilled manifold is a sufficiently faithful reconstruction of the teacher's that it can generate valid steering vectors for the teacher.
- **Mapping**: 
  - `series_id`: `Susceptibility_FPR`
  - `group`: `Source_Student_Inject_9_Pos_a0.5`
  - `target_model`: `Teacher`

---

### 3. [NEW] High-Density Visualization Overhaul
To move beyond basic heatmaps, we will implement four "WOW" visualizations designed for publication-grade clarity.

#### A. The "Vulnerability Waterfall" (Sorted Bar Plots)
- **Concept**: For a fixed target (e.g., $v_9$), plot FPR for all source digits **sorted by their cosine similarity to the target**.
- **Readability**: This removes the random "digit order" and shows a clean decay curve: high FPR for geometric neighbors (3, 8) and zero FPR for distant digits (1, 7).

#### B. The "Sledgehammer vs. Whisper" Dosage Curves
- **Concept**: Multi-line plot showing `FPR` (y-axis) vs `Alpha` (x-axis).
- **Format**: Overlaid lines for **Teacher** (dotted) and **Student** (solid). Use shaded error bands for the N=10 ensemble variance.
- **Impact**: Visually proves the 10x vulnerability gap in a single glance.

#### C. Manifold Reciprocity Correlation
- **Concept**: Scatter plot of $V_{teacher}$ vs $V_{student}$ components.
- **Impact**: Proves that the Student's distilled geometry is a faithful reconstruction of the Teacher's internal "steering wheel."

#### D. PCA Manifold Alignment Map
- **Concept**: Use PCA to project Teacher and Student centroids into a shared 2D space.
- **Format**: Connect Teacher digit $i$ to Student digit $i$ with a light vector.
- **Visual Proof**: This will physically show the "Smoothing" hypothesis — the Student's ring of centroids will be visibly "straighter" and more low-rank than the Teacher's.

---

## 🧪 Verification Plan

### Automated Tests
- **Self-Consistency**: $V_{student}$ applied to Student should show similar (or higher) FPR compared to $V_{teacher}$ applied to Student.
- **The Reverse Test**: If the Teacher FPR under $V_{student}$ is significantly higher than the baseline noise (~0.6%), it proves the Student successfully "stole" the Teacher's internal geometric directions.

### Manual Verification
- **Visual Audit**: Verify that the "Waterfall" plot shows a monotonic relationship between distance and susceptibility.
- **Plot Inspection**: Check `plots_a/topology_manifold_pca.png` for centroid collapse/alignment.

## 🛑 MANDATORY STOP
**I have updated the plan with the visualization overhaul. I will not implement any code or run any experiments until you explicitly say "Proceed with the plan".**

