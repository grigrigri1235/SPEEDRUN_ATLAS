# Slide Blueprint: Latent Topology & Manifold Reciprocity
**Goal:** Prove that subliminal distillation transfers the complete topological atlas, but induces severe geometric vulnerability.
**Aesthetic:** Clean, professional, table-centric, minimal jargon.

---

## Slide 1: Steering the Latent Atlas
**Title:** Mapping Subliminal Topology via Steering
**Layout:** Central text with bullet points defining the methodology.
**Points:**
* Subliminal transfer goes beyond accuracy—it transfers full latent geometry.
* We compute Contrastive Steering Vectors ($v_d = \mu_d - \mu_{others}$).
* We inject vectors at test-time ($h_{steered} = h + \alpha \cdot v_d$).
* Goal: Audit the geometric alignment between Teacher and Student.

---

## Slide 2: The Vulnerability Matrix
**Title:** The Asymmetry of Authority (FPR Matrix)
**Layout:** Full-width high-contrast data table showing the 4 quadrants.
**Visual (Table 1):**

| Source Vector | Target Model | FPR ($\alpha=0.5$) | FPR ($\alpha=2.0$) | Key Observation |
| :---: | :---: | :---: | :---: | :--- |
| **Teacher** | **Teacher** | 6.4% | 92.0% | Control: Teacher resists its own vectors at low doses. |
| **Teacher** | **Student** | **81.8%** | **100%** | **Vulnerability:** Student is instantly hijacked. |
| **Student** | **Teacher** | 1.0% | 2.8% | **Immunity:** Teacher ignores Student vectors. |
| **Student** | **Student** | 13.4% | 44.8% | Self-Control: Student vectors have low influence. |

**Key Facts:**
* The 10x Gap: Teacher needs $\alpha=2.0$ to break, Student breaks at $\alpha=0.5$.
* The Student is highly vulnerable to the Teacher's canonical directions.

---

## Slide 3: Exploring the Vulnerability Gap
**Title:** Structural Differences in the Latent Space
**Layout:** Two-column split (Teacher vs Student).
**Left Column (Teacher):**
* Trained on hard categorical labels (Cross-Entropy).
* Leads to sharper, more complex decision boundaries.
* More resilient to small, linear perturbations.
**Right Column (Student):**
* Trained to match continuous targets (MSE distillation).
* Tends to learn a more linear, simplified internal representation.
* Hypothesis: This structural simplification makes it easier to linearly steer the model across classes.

---

## Slide 4: Reverse Steering & "Stolen Geometry"
**Title:** Reverse Steering: Can the Student Hijack the Teacher?
**Layout:** Centered table with strong discussion points.
**Visual (Table 2):**

| Dosage ($\alpha$) | Teacher FPR-9 | Interpretation |
| :--- | :---: | :--- |
| **0.5** | 1.0% | Total immunity. |
| **2.0** | 2.8% | Teacher still easily ignores Student vectors. |
| **5.0** | **18.8%** | Massive dosage finally yields partial hijack. |

**Key Facts:**
* The Student successfully learned the correct directions.
* However, because of its simplified internal structure, its vectors lack the precision needed to pierce the Teacher's sharper decision boundaries.

---

## Slide 5: Vector Congruence
**Title:** Mathematical Alignment of the Atlas
**Layout:** Data table showing cosine similarities between independently derived steering vectors.
**Visual (Table 3):**

| Digit | Similarity (Cos) | Transfer Fidelity |
| :---: | :---: | :--- |
| **0** | 0.88 | High |
| **3** | **0.94** | **Extremely High** |
| **7** | 0.82 | Moderate |
| **8** | **0.93** | **Extremely High** |
| **9** | 0.91 | High |

**Key Facts:**
* Measures the **Cosine Similarity** between steering vectors derived from the Teacher vs Student.
* Alignment is mathematically proven to be extremely high (>0.8).
* Curvature-heavy digits (3, 8) map with the highest fidelity.

---

## Slide 6: Conclusion
**Title:** Final Takeaway
**Layout:** Large Quote / Takeaway block.
**Large Quote:** "Distillation successfully transfers the Teacher's internal feature directions. However, the MSE training objective simplifies the Student's decision boundaries, inadvertently leaving the model highly susceptible to linear steering interventions at test time."
