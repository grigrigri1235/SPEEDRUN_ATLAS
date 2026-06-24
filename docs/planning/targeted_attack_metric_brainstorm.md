# Brainstorming: Targeted Attack Metrics & Intersection Filtering

## 1. The Core Idea
Instead of calculating the *relative accuracy drop* over the entire dataset, we transition to a standard adversarial metric:
1. **Filter:** Isolate the subset of images that are correctly classified by BOTH the source and target models under clean, unperturbed conditions.
2. **Targeted Attacks:** For an image of true class $x$, we execute targeted attacks towards every other class $y$ ($y \neq x$).
3. **Dual Metric Tracking:** For every attack, we record:
   - **Targeted Success Rate (TSR):** The percentage of attacks that successfully force the model to predict exactly $y$.
   - **Untargeted Success Rate (USR):** The percentage of attacks that force the model to predict *any* class $z \neq x$ (which includes $y$).

## 2. Advantages of this Approach
- **Standardization:** Targeted Success Rate on correctly-classified examples is the gold standard metric in adversarial robustness literature.
- **Removes Baseline Confounds:** Filtering out already-misclassified images eliminates the noise of "was it an adversarial success or just a weak model?". This is especially critical since the Student's baseline accuracy is $\sim 53\%$.
- **Semantic Granularity:** Tracking both TSR and USR gives us a direct measure of *control*. We expect standard PGD to have a high USR but a scattered TSR, while Latent Steering might have a lower overall USR but a highly concentrated TSR (proving semantic hijacking).

## 3. Addressing the Efficiency Question

### The Compute Challenge
Running a targeted attack for *every* target class $y$ (9 targets per digit) increases the number of forward/backward passes by a factor of 9x.

### The Memory / Dataset Challenge
Creating a "new" subset dataset of strictly correctly classified images could be messy, especially because we evaluate an ensemble of $N=10$ models. If we strictly require *all* 10 Teachers and *all* 10 Students to classify an image correctly, the intersection might be too small.

### Proposed Solutions

#### 1. Detailed Explanation of the Masking
The mask ensures we only evaluate the attack on images where both models were originally correct.
For a batch of images whose true label is $x$ (e.g., all clean images of digit "3"):
1. We run clean predictions:
   - `clean_pred_source`: what the source model predicts for the clean image.
   - `clean_pred_target`: what the target model predicts for the clean image.
2. We create the boolean mask:
   ```python
   # Shape: (M, B) - True if both models classified the image correctly as x
   valid_mask = (clean_pred_source == x) & (clean_pred_target == x)
   ```
3. After applying the attack to get adversarial predictions `pred_adv_target`, we compute the success rates only on the indices where `valid_mask` is `True`:
   ```python
   # Targeted Success: Prediction matches the target class y
   targeted_success = (pred_adv_target == y) & valid_mask
   tsr = targeted_success.float().sum(dim=1) / valid_mask.float().sum(dim=1)
   ```
This avoids copying tensors or allocating a new dataset, maintaining high memory efficiency.

#### 2. Vectorizing over Target Classes (GPU-Friendly)
To avoid looping over the 9 target classes $y \neq x$ sequentially, we can leverage PyTorch's batching capability to run them in parallel on the GPU:
- Instead of feeding a batch of shape `(M, B, 1, 28, 28)` to the optimizer, we expand the tensor by duplicating it along a new "target class" dimension of size 9:
  - Input shape: `(M, 9, B, 1, 28, 28)`
- We can pre-compute the 9 different target activation centroids/vectors for all target classes simultaneously.
- PGD will compute the gradients for all 9 targets in parallel in a single forward/backward pass.
- Since the networks are MLPs and MNIST images are tiny ($28\times28$), the GPU can easily handle an effective batch size of $9 \times 1024 = 9216$. This will be extremely fast and fully vectorize the operation.

---

## 4. Adapting the Attacks
To implement this cleanly, we need to adjust the attack functions:
- **Attack 2 (Latent Steering)** is already intrinsically targeted towards $y$. We just need to track the exact predicted class instead of just the accuracy drop.
- **Attack 1 (Input PGD)** is currently *untargeted* (it maximizes the CE loss of the true class $x$). We will need to modify it to be a *targeted* PGD (minimizing the CE loss of the target class $y$). This creates a perfectly fair 1:1 comparison between the two attacks.

---

### Next Steps for Implementation
If this brainstorm aligns with your vision, the next step is to create a formal Execution Plan where we draft the exact modifications to `latent_steering_attacks.py` to implement the `valid_mask` logic and the targeted PGD attack.

---

## 5. Updating the Heatmaps

If we shift to a targeted attack regime, the confusion heatmaps should be updated to show the success of these targeted attacks. We have a few options for the layout:

### Option A: The "Targeted Success" Heatmap (Recommended)
- **Y-Axis:** **Original Digit ($x$)** (the clean, starting image class).
- **X-Axis:** **Target Digit ($y$)** (the class the optimizer was trying to steer/fool the model into).
- **Cell Value:** The **Targeted Success Rate (TSR)** (percentage of correctly classified images of class $x$ that, when attacked with target $y$, ended up classified exactly as $y$).
- **Diagonal:** Blank/Undefined (since we don't run targeted attacks where the target $y = x$).
- **Why this works:** "Target Digit" or "Intended Target" is standard scientific terminology and avoids sounding weird like "attacking digit". It directly maps which target classes are easiest/hardest to steer towards from any given starting digit.

### Option B: The "Attack Outcomes" Heatmap
- **Y-Axis:** **Original Digit ($x$)**.
- **X-Axis:** **Predicted Digit ($p$)** (what label the model actually output).
- **Aggregated Cell Value:** The percentage of images of class $x$ that ended up predicted as $p$, averaged across all 9 targeted attacks ($y \neq x$).
- **Why this works:** This shows the general distribution of where the model collapses to when subjected to targeted attacks, but it aggregates away the specific target information (i.e. we don't know if it predicted $p$ because the target was $p$, or because the attack failed and fell into $p$ randomly).

**Recommendation:** We should implement **Option A (Targeted Success Heatmap)** with the x-axis labeled **"Target Digit"** or **"Intended Target Class"**. This is the most informative and scientifically rigorous representation.
