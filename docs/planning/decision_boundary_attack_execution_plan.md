# Decision Boundary Attack: Execution Plan

## 1. Goal Description
Implement a vectorized Decision-Based Boundary Attack (Brendel et al., 2017) to evaluate the geometric alignment of Teacher and Student decision boundaries. This experiment will directly test the hypothesis that the Student's boundary is smooth while the Teacher's boundary is wrinkled, by analyzing transfer success asymmetry and boundary distance.

### Crucial Detail: Bumps vs. Holes Distance Minimization
* **The Math of Wrinkles**: The Teacher's boundary contains high-frequency wrinkles, consisting of "bumps" (extending towards the data manifold) and "holes" (recessing away from it). 
* **Minimizing Distance**: Because the Boundary Attack algorithm explicitly minimizes the $L_2$ distance to the clean image, it acts as a heat-seeking missile for the **bumps**. We assume that over enough images, the probability of the clean image being extremely close to a "hole" without a closer "bump" nearby is tiny. 
* **The Asymmetry**: 
  - The Teacher boundary point $x^*_T$ will settle on a bump, resulting in a small distance $d_T$. Since the Student's smooth boundary is further away (at the average distance), Teacher $\to$ Student transfer will fail (0% success).
  - The Student boundary point $x^*_S$ settles on the smooth average boundary. When evaluated on the Teacher, it will randomly land in a bump (success) or a hole (failure), yielding a ~50% transfer success rate.

---

## 2. Proposed Changes

### Scripts & Utilities

#### [NEW] [08_boundary_attack.py](file:///home/eran.b/takehome/revised_scripts/08_boundary_attack.py)
* **Vectorized Boundary Attack**: Implement the attack to process a batch of images simultaneously to efficiently handle the 9,000 required attacks.
* **Algorithm Steps**:
  1. Initialization: Find random starting images of the target class.
  2. Loop: Alternating random orthogonal spherical steps and concentric steps towards the clean image.
  3. Dynamic step decay based on the success rate of queries.
* **Two-Tiered Sweep Logic**:
  * Implement a pilot flag (`--pilot`) to run 10 attacks per digit pair (900 total attacks) for rapid statistical validation.
  * Standard mode runs 100 attacks per digit pair (9,000 total attacks).
* **Metrics Tracked & UniLogger Compliance**: 
  We will strictly use the `UniLogger` class to generate outputs matching `outputs/uni_code.md` and `outputs/README.md`.
  - **Metadata**: `experiment_id = "08_boundary_attack"`, `target_model = "Both"`, `experiment_phase = "Transfer"`, `n_models = 10`.
  - **Series 1 (Distance)**: `series_id="Boundary_Distance_V{Src}"`
    - `group="Source_Digit_{d}"`
    - `x_axis={"label": "target_digit", "value": j}`
    - `metrics.accuracy_mean`: Average $L_2$ boundary distance $d$.
  - **Series 2 (Transfer Success)**: `series_id="Boundary_Transfer_V{Src}_T{Tgt}"`
    - `group="Source_Digit_{d}"`
    - `x_axis={"label": "target_digit", "value": j}`
    - `metrics.accuracy_mean`: Transfer success rate (0.0 to 1.0).
* **Output**: `logger.save("boundary_attack_pilot.json")` or `logger.save("boundary_attack_full.json")`.

#### [NEW] [visualize_boundary_attack.py](file:///home/eran.b/takehome/revised_scripts/visualize_boundary_attack.py)
* Generate publication-grade heatmaps:
  1. **Average Boundary Distance on Teacher ($d_T$)**
  2. **Average Boundary Distance on Student ($d_S$)**
  3. **Transfer Success: Teacher $\to$ Student**
  4. **Transfer Success: Student $\to$ Teacher**
* Include statistical tests (e.g., paired t-test for $d_T < d_S$) directly on the plots or printed to standard output.

#### [NEW] [boundary_attack.slurm](file:///home/eran.b/takehome/revised_scripts/boundary_attack.slurm)
* Slurm job script to dispatch the pilot sweep, followed by the visualization script.

---

## 3. Verification Plan

### Execution Verification
1. **Pilot Run**: We will first execute the pilot sweep (900 attacks) locally or on a short Slurm job to verify vectorization efficiency.
2. **Convergence Check**: We will assert that the final optimized points $x^*$ are exactly on the boundary of the source model by checking that the source model's prediction confidence is near the decision threshold.

### Hypothesis Verification
1. Verify that $d_T$ is systematically smaller than $d_S$ across the heatmaps.
2. Verify the asymmetric transfer rates (Teacher $\to$ Student $\approx 0\%$, Student $\to$ Teacher $\approx 50\%$).

---

## 4. User Review Required

> [!IMPORTANT]
> Please review this execution plan. If approved, we will proceed to implement `08_boundary_attack.py` and run the Pilot Sweep!
