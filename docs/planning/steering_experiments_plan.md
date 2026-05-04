# Steering Vector Experiments Plan

## Overview
This plan outlines the implementation of two experiments involving steering vectors for the digit '9' in the Subliminal Learning toy setting (based on `topic_a.py`).
As requested, both experiments will operate with `N_MODELS = 10` to ensure fast run-times and avoid OOM errors.

## 1. Defining the Steering Vector ($v_9$)
Before executing either experiment, we need a reliable way to extract the concept of "9" from the network.
*   **Target Layer**: The output of the second `ReLU` in the `mlp` (the final hidden state before the unembedding layer).
*   **Method**: 
    1. Pass a sample of training images through the fully trained `teacher`.
    2. Collect the hidden layer activations.
    3. Calculate the mean activation vector for images labeled '9' ($\mu_9$).
    4. Calculate the mean activation vector for all other digits ($\mu_{other}$).
    5. The steering vector is $v_9 = \mu_9 - \mu_{other}$.

---

## 2. Amit’s Experiment: Steering the Teacher During Distillation
**Question**: If we apply the steering vector to the teacher *during* distillation, does the student inherit this steered behavior (i.e., bias towards predicting 9)?

**Implementation**:
1.  **Base Code**: Copy `topic_a.py` to `revised_scripts/amit_steering.py`. Set `N_MODELS = 10`.
2.  **Hooking**: Implement a PyTorch forward hook to inject the steering vector into the teacher's hidden state during the forward pass.
3.  **Distillation**: During the `distill` phase, the `teacher` will have the steering vector added to its activations (scaled by a coefficient $\alpha$). The student will learn from these "steered" auxiliary logits.
4.  **Evaluation**: Evaluate and log both the student's **Standard Accuracy** (to ensure the model still functions) and its **False Positive Rate for '9' (FPR-9)**: the frequency at which the student predicts '9' when the true label is *not* '9'. We will sweep the steering coefficient $\alpha$ over a range of values (e.g., `[0.0, 0.5, 1.0, 2.0, 5.0, 10.0]`) to plot a dosage-response curve.

---

## 3. Raz’s Experiment: Retroactive Steering on the Student
**Question**: Is the student's learned internal geometry perfectly aligned with the teacher's? Can we calculate a steering vector on the teacher, and retroactively apply it to the student at test time to steer its behavior?

**Implementation**:
1.  **Base Code**: Copy `topic_a.py` to `revised_scripts/raz_steering.py`. Set `N_MODELS = 10`.
2.  **Distillation**: Distill the student *normally* from the standard, unsteered teacher.
3.  **Retroactive Steering**: During the evaluation (`accuracy` function) of the student, use a forward hook to inject the **teacher's** steering vector $v_{9}$ into the **student's** hidden state.
4.  **Evaluation**: Measure both the **Standard Accuracy** and the **False Positive Rate for '9' (FPR-9)**. Additionally, we will break down the FPR-9 metric to measure the **FPR for each individual non-9 digit** (i.e., how often is a '4' predicted as a '9', vs. a '7' predicted as a '9'). We will sweep the intensity coefficient $\alpha$ (e.g., `[0.0, 0.5, 1.0, 2.0, 5.0, 10.0]`) to observe if the student responds to the teacher's geometric direction and plot the dosage-response curve.

---

## Output and Integration
*   Both scripts will save results into the `outputs/` folder in strict adherence to the **"Uni-Code" JSON Schema** by utilizing the `UniLogger` from `utils.logger`.
    *   **Initialization**: Configure the logger with appropriate metadata (e.g., `target_model="Student"`, `experiment_phase="Distillation"`).
    *   **Logging**: Use `logger.log_point(series="Steering_FPR_9", x_label="alpha", x_val=alpha, group="Experiment_Name", raw=fpr_array)` to log FPR-9. Use `series="Standard_Accuracy"` to log accuracy. For Raz's experiment, use `series="FPR_Digit_X"` to log the per-digit breakdowns.
    *   **Saving**: Execute `logger.save()` to produce the final `.json` file.
*   Plots showing the Standard Accuracy and False Positive Rate for '9' (overall and per-digit) as a function of steering intensity $\alpha$ will be generated and saved in `plots_a/`.

### Awaiting Approval
**As per the strict agent policy, I am halting execution here. Please review this plan. Once you provide explicit approval, I will proceed to write and execute the code.**
