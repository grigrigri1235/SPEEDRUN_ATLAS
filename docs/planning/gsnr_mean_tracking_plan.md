# Implementation Plan: GSNR Mean Tracking

## Overview
The goal of this task is to track the mean of the weight changes alongside the variance in `dropout_analysis_sweep.py`. This aligns with the Gradient Signal-to-Noise Ratio (GSNR) theory, where GSNR is generally a function of both the mean signal and the variance of the noise. The updates will also be documented in `outputs/README.md`.

## Changes to `revised_scripts/dropout_analysis_sweep.py`

1. **Rename and Update Function**:
   - Rename `get_weight_change_variance` to `get_weight_change_stats` to better reflect that it will return both variance and mean.
   - Update the internal logic to calculate the mean of the weight changes for both MNIST and Ghost channels.
   - Add new keys to the returned dictionary: `Layer2_Weight_Change_Mean_MNIST` and `Layer2_Weight_Change_Mean_Ghost`.

2. **Update Function Calls**:
   - In `run_experiment`, change the calls from `get_weight_change_variance` to `get_weight_change_stats`.
   - The returned variables `t_var` and `s_var` can be renamed to `t_change_stats` and `s_change_stats` for clarity.

3. **Update Data Aggregation in `main`**:
   - The `log_all` function naturally iterates over the dictionaries, so the new metrics will automatically be logged by `UniLogger`.
   - Update the plotting data dictionary (`pt`) to include:
     - `s_mean_ghost_mean`: `float(np.mean(res['s_change_stats']['Layer2_Weight_Change_Mean_Ghost']))`
     - `s_mean_ghost_std`: `float(np.std(res['s_change_stats']['Layer2_Weight_Change_Mean_Ghost']))`
     - `t_mean_ghost_mean`: `float(np.mean(res['t_change_stats']['Layer2_Weight_Change_Mean_Ghost']))`
     - `t_mean_ghost_std`: `float(np.std(res['t_change_stats']['Layer2_Weight_Change_Mean_Ghost']))`
   - Update variable names in `log_all` from `t_var` and `s_var` to `t_change_stats` and `s_change_stats`.

## Changes to `outputs/README.md`

1. **Document New Metrics**:
   - Under the **Phase 1: Regularization & Internal Mechanics** section (item 4: Dropout Sweep), add documentation for the newly tracked variables.
   - Specify that `Teacher_Layer2_Weight_Change_Mean_MNIST` / `_Ghost` and `Student_Layer2_Weight_Change_Mean_MNIST` / `_Ghost` track the numerator (mean signal) of the Gradient Signal-to-Noise Ratio (GSNR).
   - This ensures the README remains an accurate, friendly, and complete mapping guide for future AI agents.

## Understanding of GSNR Theory & Logical Assessment

**What is the GSNR Theory in this Context?**
The Gradient Signal-to-Noise Ratio (GSNR) dictates whether a neural network can successfully learn a feature. Mathematically, $\text{GSNR} \approx \frac{\mu^2}{\sigma^2}$, where $\mu$ is the expected gradient (the true "Signal") and $\sigma^2$ is the variance of the gradient (the stochastic "Noise"). 
In the context of the Subliminal Learning experiment, the "Ghost" channel carries a very weak subliminal signal ($\mu$ is small). When internal dropout noise is applied to the Student, it injects massive multiplicative variance ($\sigma^2$) into the gradients. Because the ghost signal is weak, this added noise causes the GSNR to collapse below 1.0, turning the learning process for the ghost channel into a random walk. Conversely, the primary MNIST task has a massive signal ($\mu$ is large), so its GSNR remains healthy even with dropout.

**Is it logical to track the mean?**
Yes, it is extremely logical and necessary. Currently, the codebase only tracks the variance of the weight change ($\text{Var}(W_{\text{final}} - W_{\text{init}})$ across $N=10$ runs) as an empirical proxy for the noise ($\sigma^2$). However, measuring *only* the noise doesn't prove a GSNR collapse—you must also measure the signal to show that the noise overwhelms it. By tracking the **mean weight change** ($\mathbb{E}[W_{\text{final}} - W_{\text{init}}]$) across the ensemble, we explicitly measure the directed "Signal" ($\mu$). 
Having both the empirical mean and variance will allow us to directly compute the empirical GSNR proxy ($\frac{\text{Mean}^2}{\text{Variance}}$) and definitively prove mathematically that the signal is drowned out by the dropout noise for the Ghost channel but not for the MNIST channel.

## Next Steps
Upon your approval, I will execute these code and documentation changes.
