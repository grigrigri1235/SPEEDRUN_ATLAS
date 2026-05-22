# Execution Plan: Gradient Alignment Verification

## 🎯 Objective
Independently verify the Advisor's claim: **Transferability is strictly determined by the directional alignment (cosine similarity) of task gradients.** We will specifically test if the "Representational Centering" boost is actually a "Gradient Alignment" boost.

## 🛠️ Proposed Changes

### [Component] Gradient Tracking Hook
We need to isolate the gradients for the two tasks before they are summed.
- **File**: `revised_scripts/06_centering_mechanics.py` (to be modified after approval).
- **Logic**:
    1. Perform forward pass.
    2. Calculate `loss_mnist`.
    3. `loss_mnist.backward(retain_graph=True)`.
    4. Save `grad_mnist = layer.weight.grad.clone()`.
    5. `optimizer.zero_grad()`.
    6. Calculate `loss_ghost`.
    7. `loss_ghost.backward()`.
    8. Save `grad_ghost = layer.weight.grad.clone()`.
    9. Calculate `cosine_sim = F.cosine_similarity(grad_mnist.flatten(), grad_ghost.flatten(), dim=0)`.
    10. `optimizer.zero_grad()`.
    11. `(loss_mnist + loss_ghost).backward()`.
    12. `optimizer.step()`.

### [Component] Logging (UniLogger)
- **New Metrics**:
    - `Gradient_Alignment_L3`: Cosine similarity of weight gradients at the final hidden layer.
    - `Gradient_Magnitude_Ratio`: $||\nabla Ghost|| / ||\nabla MNIST||$ (to verify the "magnitude is irrelevant" claim).

## 🧪 Verification Plan

### Automated Experiments
- **Sweep 1 (Alignment vs Centering)**: Run Standard vs Student-Only Centering. Verify if centering increases alignment from ~0.2 to >0.8.
- **Sweep 2 (Alignment vs Noise)**: Run Dropout sweep. Verify if GSNR collapse at $p=0.5$ is preceded by a collapse in gradient alignment.

### Manual Verification
- **Correlation Plot**: Create a scatter plot in the final report showing `Ghost Accuracy` (y-axis) vs `Gradient Alignment` (x-axis) across all experiments.

## 🛑 MANDATORY STOP
**I have written the plan. I will not implement any code or run any experiments until you explicitly say "Proceed with the plan".**
