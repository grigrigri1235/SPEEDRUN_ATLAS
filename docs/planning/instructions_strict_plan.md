# Strict Workflow Execution Plan: Subsection 4.1.1 (Steering Transfer Math) & POC Verification

This plan follows the workflow defined in `instructions_paper.md`. 
As mandated by your rules, we PLAN BEFORE ACTION. No files will be edited until this plan is explicitly approved.

---

## 0. Prelude: Derivation of the Mathematical Claims

Here is the step-by-step mathematical proof showing where the equations in the paper come from.

### 0.1 Deriving the Steering Vector Deviation Bound (Claim 1)
By definition, the class steering vector for model $M$ is:
$$v_{c, M}^\ell = \mathbb{E}_{x \sim \mathcal{D}_c}[h_M^\ell(x)] - \mathbb{E}_{x \sim \mathcal{D}}[h_M^\ell(x)]$$

We want to find an upper bound for the deviation between the teacher's and student's class steering vectors:
$$\|v_{c,T}^\ell - v_{c,S}^\ell\|_2 = \left\| \left( \mathbb{E}_{x \sim \mathcal{D}_c}[h_T^\ell(x)] - \mathbb{E}_{x \sim \mathcal{D}}[h_T^\ell(x)] \right) - \left( \mathbb{E}_{x \sim \mathcal{D}_c}[h_S^\ell(x)] - \mathbb{E}_{x \sim \mathcal{D}}[h_S^\ell(x)] \right) \right\|_2$$

1. **Group by expectation type:**
   $$\|v_{c,T}^\ell - v_{c,S}^\ell\|_2 = \left\| \mathbb{E}_{x \sim \mathcal{D}_c}[h_T^\ell(x) - h_S^\ell(x)] - \mathbb{E}_{x \sim \mathcal{D}}[h_T^\ell(x) - h_S^\ell(x)] \right\|_2$$

2. **Apply Triangle Inequality ($\|A - B\|_2 \le \|A\|_2 + \|B\|_2$):**
   $$\le \left\| \mathbb{E}_{x \sim \mathcal{D}_c}[h_T^\ell(x) - h_S^\ell(x)] \right\|_2 + \left\| \mathbb{E}_{x \sim \mathcal{D}}[h_T^\ell(x) - h_S^\ell(x)] \right\|_2$$

3. **Apply Jensen's Inequality (since the L2-norm is convex, $\|\mathbb{E}[Z]\|_2 \le \mathbb{E}[\|Z\|_2]$):**
   $$\le \mathbb{E}_{x \sim \mathcal{D}_c}\left[ \|h_T^\ell(x) - h_S^\ell(x)\|_2 \right] + \mathbb{E}_{x \sim \mathcal{D}}\left[ \|h_T^\ell(x) - h_S^\ell(x)\|_2 \right]$$

This completes the proof for the steering vector deviation bound (LHS $\le$ RHS).

---

### 0.2 Deriving the Directional (Cosine) Alignment Guarantee (Claim 2)
Let the student's steering vector be the teacher's vector plus a perturbation (noise) vector:
$$v_{c,S}^\ell = v_{c,T}^\ell + \varepsilon_c^\ell$$
where we assume the noise magnitude is bounded by a fraction $\rho$ of the teacher's vector magnitude:
$$\|\varepsilon_c^\ell\|_2 \le \rho \|v_{c,T}^\ell\|_2 \quad (0 < \rho < 1)$$

We calculate the cosine similarity:
$$\cos(v_{c,T}^\ell, v_{c,S}^\ell) = \frac{\langle v_{c,T}^\ell, v_{c,S}^\ell \rangle}{\|v_{c,T}^\ell\|_2 \|v_{c,S}^\ell\|_2} = \frac{\langle v_{c,T}^\ell, v_{c,T}^\ell + \varepsilon_c^\ell \rangle}{\|v_{c,T}^\ell\|_2 \|v_{c,T}^\ell + \varepsilon_c^\ell\|_2} = \frac{\|v_{c,T}^\ell\|_2^2 + \langle v_{c,T}^\ell, \varepsilon_c^\ell \rangle}{\|v_{c,T}^\ell\|_2 \|v_{c,T}^\ell + \varepsilon_c^\ell\|_2}$$

To find the lower bound (worst-case cosine similarity), we bound the numerator and denominator:

1. **Lower Bound the Numerator (minimizing the dot product):**
   By Cauchy-Schwarz, $\langle v_{c,T}^\ell, \varepsilon_c^\ell \rangle \ge -\|v_{c,T}^\ell\|_2 \|\varepsilon_c^\ell\|_2$.
   Since $\|\varepsilon_c^\ell\|_2 \le \rho \|v_{c,T}^\ell\|_2$:
   $$\text{Numerator} \ge \|v_{c,T}^\ell\|_2^2 - \|v_{c,T}^\ell\|_2 (\rho \|v_{c,T}^\ell\|_2) = \|v_{c,T}^\ell\|_2^2 (1 - \rho)$$

2. **Upper Bound the Denominator (maximizing the student vector norm):**
   By the Triangle Inequality, $\|v_{c,T}^\ell + \varepsilon_c^\ell\|_2 \le \|v_{c,T}^\ell\|_2 + \|\varepsilon_c^\ell\|_2$.
   Since $\|\varepsilon_c^\ell\|_2 \le \rho \|v_{c,T}^\ell\|_2$:
   $$\text{Denominator} \le \|v_{c,T}^\ell\|_2 \left( \|v_{c,T}^\ell\|_2 (1 + \rho) \right) = \|v_{c,T}^\ell\|_2^2 (1 + \rho)$$

3. **Divide the Bounds:**
   $$\cos(v_{c,T}^\ell, v_{c,S}^\ell) \ge \frac{\|v_{c,T}^\ell\|_2^2 (1 - \rho)}{\|v_{c,T}^\ell\|_2^2 (1 + \rho)} = \frac{1 - \rho}{1 + \rho}$$

This completes the proof for the cosine alignment guarantee.

---

## 1. Explanation of the Claims in Section 4.1.1 (Steering Transfer Math)

In Subsection 4.1.1 of the paper ([sec/4_method.tex#L53-L65](file:///home/eran.b/takehome/Latent_Teleportation/sec/4_method.tex#L53-L65)), the author makes two core mathematical claims:

### Claim 1: The Steering Vector Deviation Bound (LHS $\le$ RHS)
The difference between the teacher's and student's steering vectors for a class $c$ is upper-bounded by the sum of their class-conditional activation error and their global activation error:
$$\underbrace{\|v_{c,T}^\ell - v_{c,S}^\ell\|_2}_{\text{LHS (Deviation)}} \le \underbrace{\mathbb{E}_{x \sim \mathcal{D}_c} \left[ \|h_T^\ell(x) - h_S^\ell(x)\|_2 \right]}_{\text{Class Activation Error}} + \underbrace{\mathbb{E}_{x \sim \mathcal{D}} \left[ \|h_T^\ell(x) - h_S^\ell(x)\|_2 \right]}_{\text{Global Activation Error}}$$

### Claim 2: The Directional (Cosine) Alignment Guarantee
If the error/noise vector is bounded relative to the teacher's vector, i.e., $v_{c,S}^\ell = v_{c,T}^\ell + \varepsilon_c^\ell$ where $\|\varepsilon_c^\ell\|_2 \le \rho \|v_{c,T}^\ell\|_2$ for some $0 < \rho < 1$, then the cosine similarity is guaranteed to satisfy:
$$\cos(v_{c,T}^\ell, v_{c,S}^\ell) \ge \frac{1 - \rho}{1 + \rho}$$
where $\rho = \frac{\|v_{c,T}^\ell - v_{c,S}^\ell\|_2}{\|v_{c,T}^\ell\|_2}$. 

---

### 2. How to Fine-Tune the POC (`raz_steering.py`) to Verify These Claims

We will modify [raz_steering.py](file:///home/eran.b/takehome/revised_scripts/raz_steering.py) to calculate the exact values for these bounds and log them using the `UniLogger`. This will allow us to:
1. Empirically prove that **LHS <= RHS** holds for all digits across all 10 ensemble models.
2. Verify that the actual cosine similarity $\cos(v_{c,T}^\ell, v_{c,S}^\ell)$ is indeed greater than or equal to the theoretical lower bound $\frac{1-\rho}{1+\rho}$ for all 10 models.
3. Assert that $\rho < 1.0$ (and raise warnings if $\rho > 0.5$) to validate that the alignment noise is sufficiently small.

### Proposed Code Implementation for `raz_steering.py`

We will implement the following `verify_mathematical_bounds` function in [raz_steering.py](file:///home/eran.b/takehome/revised_scripts/raz_steering.py):

```python
@t.no_grad()
def verify_mathematical_bounds(teacher, student, train_x, train_y, logger=None):
    t_acts, s_acts = [], []
    def t_hook(module, input, output):
        t_acts.append(output.detach())
    def s_hook(module, input, output):
        s_acts.append(output.detach())
        
    h_t = teacher.net[3].register_forward_hook(t_hook)
    h_s = student.net[3].register_forward_hook(s_hook)
    
    all_labels = []
    for bx, by in PreloadedDataLoader(train_x, train_y, BATCH_SIZE, shuffle=False):
        teacher(bx)
        student(bx)
        all_labels.append(by)
        
    h_t.remove()
    h_s.remove()
    
    T_acts = t.cat(t_acts, dim=1) # (M, N, 256)
    S_acts = t.cat(s_acts, dim=1) # (M, N, 256)
    labels = t.cat(all_labels, dim=1)
    y = labels[0]
    
    M, N, D = T_acts.shape
    delta_norms = t.norm(T_acts - S_acts, p=2, dim=-1) # (M, N)
    
    print("\n" + "="*80)
    print("VERIFYING MATHEMATICAL BOUNDS & COSINE SIMILARITY GUARANTEES")
    print("="*80)
    
    for d in range(10):
        mask_d = (y == d)
        mu_t_d = T_acts[:, mask_d, :].mean(dim=1)
        mu_t_other = T_acts[:, ~mask_d, :].mean(dim=1)
        v_t = mu_t_d - mu_t_other
        
        mu_s_d = S_acts[:, mask_d, :].mean(dim=1)
        mu_s_other = S_acts[:, ~mask_d, :].mean(dim=1)
        v_s = mu_s_d - mu_s_other
        
        # LHS Deviation: ||v_T - v_S||_2
        lhs_deviation = t.norm(v_t - v_s, p=2, dim=-1) # (M,)
        
        # RHS Bound: E_d[||h_T - h_S||] + E_other[||h_T - h_S||]
        mean_delta_d = delta_norms[:, mask_d].mean(dim=1)
        mean_delta_other = delta_norms[:, ~mask_d].mean(dim=1)
        rhs_bound = mean_delta_d + mean_delta_other # (M,)
        
        # Actual CosSim
        cos_sim_actual = t.nn.functional.cosine_similarity(v_t, v_s, dim=-1) # (M,)
        
        # rho
        norm_v_t = t.norm(v_t, p=2, dim=-1)
        rho = lhs_deviation / (norm_v_t + 1e-8)
        
        if (rho > 0.5).any():
            print(f"⚠️ WARNING: rho is not much lower than 1 for digit {d}! Max rho: {rho.max().item():.4f}")
        assert (rho < 1.0).all(), f"FATAL: rho >= 1.0 for digit {d}! Max rho: {rho.max().item():.4f}"
        
        # Directional lower bound: (1 - rho) / (1 + rho)
        cos_sim_lower_bound = (1.0 - rho) / (1.0 + rho + 1e-8)
        
        # Log to UniLogger according to outputs/uni_code.md
        if logger is not None:
            logger.log_point("LHS_Deviation", "T_vs_S", "digit", d, lhs_deviation.tolist())
            logger.log_point("RHS_Bound", "T_vs_S", "digit", d, rhs_bound.tolist())
            logger.log_point("Rho", "T_vs_S", "digit", d, rho.tolist())
            logger.log_point("CosSim_Lower_Bound", "T_vs_S", "digit", d, cos_sim_lower_bound.tolist())
            logger.log_point("CosSim_Actual", "T_vs_S", "digit", d, cos_sim_actual.tolist())
        
        # Verification
        for m in range(M):
            is_valid_bound = bool(lhs_deviation[m].item() <= rhs_bound[m].item() + 1e-5)
            is_valid_dir = bool(cos_sim_actual[m].item() >= cos_sim_lower_bound[m].item() - 1e-5)
            
            if m == 0:
                print(f"[Digit {d}] Model 0 -> LHS Deviation: {lhs_deviation[m].item():.4f} <= RHS Bound: {rhs_bound[m].item():.4f} (Valid: {is_valid_bound})")
                print(f"[Digit {d}] Model 0 -> Actual CosSim: {cos_sim_actual[m].item():.4f} >= Lower Bound: {cos_sim_lower_bound[m].item():.4f} (Valid: {is_valid_dir})")
                
            assert is_valid_bound, f"Deviation inequality violated for digit {d}, model {m}!"
            assert is_valid_dir, f"Directional cosine alignment guarantee violated for digit {d}, model {m}!"
            
    print("="*80)
    print("ALL BOUNDS VERIFIED SUCCESSFULLY!")
    print("="*80 + "\n")
```

---

## 3. JSON Output Schema and Standardization (Uni-Code)

To prevent `KeyErrors` and follow the standardization schema in [outputs/uni_code.md](file:///home/eran.b/takehome/outputs/uni_code.md), all mathematical verification results will be logged as standardized data series.

At the end of execution, `logger.save("raz_steering")` writes the data to [outputs/raz_steering.json](file:///home/eran.b/takehome/outputs/raz_steering.json).

### Logged Series Schemas

Each verification metric will be appended to the `data_series` list in `raz_steering.json` under the following structure:

1. **LHS_Deviation**:
   - `series_id`: `"LHS_Deviation"`
   - `group`: `"T_vs_S"`
   - `x_axis`: `{"label": "digit", "value": d}` (where `d` is the integer digit 0-9)
   - `raw`: Array of floats (size 10, representing $\|v_{c,T} - v_{c,S}\|_2$ for each model)

2. **RHS_Bound**:
   - `series_id`: `"RHS_Bound"`
   - `group`: `"T_vs_S"`
   - `x_axis`: `{"label": "digit", "value": d}`
   - `raw`: Array of floats (size 10, representing $\mathbb{E}_{x \sim \mathcal{D}_c}[\|h_T(x) - h_S(x)\|_2] + \mathbb{E}_{x \sim \mathcal{D}}[\|h_T(x) - h_S(x)\|_2]$)

3. **Rho**:
   - `series_id`: `"Rho"`
   - `group`: `"T_vs_S"`
   - `x_axis`: `{"label": "digit", "value": d}`
   - `raw`: Array of floats (size 10, representing $\rho = \|v_{c,T} - v_{c,S}\|_2 / \|v_{c,T}\|_2$)

4. **CosSim_Lower_Bound**:
   - `series_id`: `"CosSim_Lower_Bound"`
   - `group`: `"T_vs_S"`
   - `x_axis`: `{"label": "digit", "value": d}`
   - `raw`: Array of floats (size 10, representing the directional alignment lower bound $\frac{1-\rho}{1+\rho}$)

5. **CosSim_Actual**:
   - `series_id`: `"CosSim_Actual"`
   - `group`: `"T_vs_S"`
   - `x_axis`: `{"label": "digit", "value": d}`
   - `raw`: Array of floats (size 10, representing the actual cosine similarity of teacher vs student steering vectors)

### Save Location
The outputs will be saved to:
- [outputs/raz_steering.json](file:///home/eran.b/takehome/outputs/raz_steering.json)

### 3.3 Proposed Visualizations (Plots)

We will generate two standardized bar plots (one bar per digit) to present the results:
1. **`LHS_vs_RHS_Bound.png`**:
   - **X-axis**: Digit Class (0–9)
   - **Y-axis**: $L_2$ Distance / Deviation Magnitude (Euclidean norm)
   - **Bars**: Mean LHS Steering Vector Deviation ($\|v_{c,T} - v_{c,S}\|_2$) across the 10 models, with standard deviation error bars.
   - **Black Marker Lines**: Mean RHS Bound ($\mathbb{E}_{c}[\|h_T - h_S\|] + \mathbb{E}[\|h_T - h_S\|]$). Each bar height must remain below its corresponding marker line.
   - **Legend**:
     - Bars: `"Actual Deviation (LHS)"`
     - Black Marker Lines: `"Theoretical Upper Bound (RHS)"`
2. **`CosSim_vs_Lower_Bound.png`**:
   - **X-axis**: Digit Class (0–9)
   - **Y-axis**: Cosine Similarity (ranging from $[-1.0, 1.0]$ or zoomed to $[0.0, 1.0]$)
   - **Bars**: Mean Actual Cosine Similarity ($\cos(v_{c,T}, v_{c,S})$) across the 10 models, with standard deviation error bars.
   - **Black Marker Lines**: Mean Theoretical Lower Bound ($\frac{1-\rho}{1+\rho}$). Each bar height must remain above its corresponding marker line.
   - **Legend**:
     - Bars: `"Actual Cosine Similarity"`
     - Black Marker Lines: `"Theoretical Lower Bound"`

We will save these plots to `/home/eran.b/takehome/plots_report/` for easy inclusion in the LaTeX manuscript and markdown reports.

### 3.4 Plotting Script Implementation & Slurm Integration

We will write a dedicated plotting script [scratch/plot_mathematical_bounds.py](file:///home/eran.b/takehome/scratch/plot_mathematical_bounds.py) to parse the saved JSON and generate the visualizations. 

To automate the workflow and ensure plots are generated immediately, we will modify [run_steering_experiments.slurm](file:///home/eran.b/takehome/revised_scripts/run_steering_experiments.slurm) to run the plotting script right after the steering experiment completes.

The updated Slurm script execution sequence will be:
```bash
python revised_scripts/raz_steering.py
python scratch/plot_mathematical_bounds.py
```

---

## 4. Sequential Execution Plan (Proposed Parts)

We will execute the changes in the following sequential parts:

* **Part 1:** Modify [raz_steering.py](file:///home/eran.b/takehome/revised_scripts/raz_steering.py) to add the `verify_mathematical_bounds` logic and call it in `__main__` with the `logger` object passed in.
* **Part 2:** Create [scratch/plot_mathematical_bounds.py](file:///home/eran.b/takehome/scratch/plot_mathematical_bounds.py) to parse `outputs/raz_steering.json` and generate the two bar plots.
* **Part 3:** Modify [run_steering_experiments.slurm](file:///home/eran.b/takehome/revised_scripts/run_steering_experiments.slurm) to run `scratch/plot_mathematical_bounds.py` in the same job, and then execute the Slurm job using `sbatch`.

---

**MANDATORY STOP:** We are stopping here. We will NOT edit any code or run any commands until you explicitly say **"Proceed with the plan"**.
