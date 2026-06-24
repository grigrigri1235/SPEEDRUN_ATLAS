# Decision-Based & Genetic Attacks Planning

## 1. Comparing the Two Gradient-Free Attacks

We compare the two main black-box/gradient-free approaches to probe the decision boundaries:

### Option A: The Boundary Attack (Brendel et al., 2017)
* **How it works**: It starts with a real image of the target class (which is adversarial) and walks along the decision boundary towards the original image. It does this by making random orthogonal steps (to stay on the boundary) and concentric steps (to get closer to the original image).
* **Pros**: It is guaranteed to find the *closest point on the decision boundary* to the clean image. This allows us to measure the exact geometric distance to the boundary.
* **Cons**: Can take many iterations (e.g., 500-1000 steps per sample) to converge.

### Option B: Genetic / Evolutionary Attack (Alzantot et al.)
* **How it works**: It starts with the original clean image. It creates a population of random perturbations (mutations). In each generation, it evaluates which perturbations increase the target class probability the most (fitness), and breeds them (crossover + mutation) to find a successful perturbation.
* **Pros**: Highly intuitive, works directly by adding noise to the clean image, and finds a perturbation that minimizes loss.
* **Cons**: Does not guarantee finding the *closest* boundary point; it just finds a perturbation within a search space that flips the label. It can also be computationally slow due to evaluating a large population.

---

## 2. In-Depth Elaboration on Option A: The Boundary Attack

### The "Finding the Boundary Wall" Analogy
Imagine you are at your house (**Clean Image $x$**, class "3"), and you want to find the **closest point on the boundary wall** that separates your yard from your neighbor's yard (**Target Class $g$**, class "8").

* **The Problem with Starting at Your House**: If you start at your house and try to find the wall by taking random steps in the dark (which is what standard black-box search does), it is highly inefficient because the space is huge (784 dimensions) and you don't know which direction to walk.
* **The Boundary Attack Solution**:
  1. **Start in the neighbor's yard**: We start at a real image of an "8" (the starting target image). We know for sure this is on the neighbor's side of the boundary.
  2. **Walk towards your house**: We take a step directly towards your house.
  3. **Check the boundary**: We query the model: "Am I still in the neighbor's yard (is the prediction still '8')?"
     * If **Yes**: We accept the step and keep moving closer.
     * If **No** (we crossed back to class "3"): We reject the step, step back to where we were, and take a **sideways step** (orthogonal random walk) to find a different angle.
  4. **Result**: By repeating this (stepping closer, adjusting sideways to stay just barely on the neighbor's side of the boundary), we slide along the boundary wall until we are standing at the **exact closest point on the boundary wall** to your house.

---

## 3. How the Algorithm Works (Clarifications)

### A. How do we know we are stepping towards the boundary?
* **The Destination**: The destination is always the **original clean image $x_{\text{orig}}$**. 
* **The Concentric Direction**: The direction we step is always the straight line towards $x_{\text{orig}}$ (concentric step).
* **The Sideways Exploration**: Since the boundary is not a straight line, we also take a random orthogonal/spherical step (sideways). This is a step along the sphere of radius $d(x^{(k)}, x_{\text{orig}})$. It keeps the distance to the original image constant but changes the direction from which we approach it, helping us find the "opening" in the boundary wall that is closest to our clean image.
* **The Decision Query**: The model's classification acts as our compass. If a step towards $x_{\text{orig}}$ is classified as the target class $g$, we are still on the neighbor's side of the boundary. If it flips to another class, we know we crossed the boundary wall, so we reject the step.

### B. When do we stop?
We stop the algorithm under two conditions:
1. **Max Iterations reached**: Usually, 500 to 1000 steps are sufficient for the distance to converge.
2. **Minimal Progress / Step Size Decay**: We dynamically shrink the step sizes ($\delta_{\text{concentric}}$ and $\delta_{\text{orth}}$) if steps keep getting rejected. Once these step sizes drop below a tiny threshold (e.g., $10^{-6}$), it means we cannot get any closer to $x_{\text{orig}}$ without crossing the boundary wall. At this point, we have converged to the boundary.
