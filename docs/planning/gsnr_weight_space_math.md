# Mathematical Definition: Intra-Run Gradient GSNR

To ensure the GSNR calculation is mathematically sound and accounts for the unaligned nature of the 10-model ensemble, we measure the GSNR **intra-run** using mini-batch stochasticity.

## 1. The Gradient GSNR (Per-Run)
We measure the GSNR of the gradients ($\nabla_\theta L$) specifically during the distillation phase. 

- **Formula (Per Run)**: 
$$GSNR_{run} = \frac{||\mathbb{E}_B[\nabla_\theta L]||^2}{Var_B(\nabla_\theta L)}$$
  Where $\mathbb{E}_B$ and $Var_B$ are taken over a large batch $B$ of samples (using per-sample gradients to capture the sampling noise).

- **Variable**: `Student_Layer2_GSNR_Ghost`
- **Definition**: The squared norm of the average gradient divided by the variance of the gradients across the batch.
- **Aggregation**: We calculate $GSNR_{run}$ for each of the $N=10$ models independently. The final reported value is the **mean** and **standard deviation** across these 10 independent results.

## 2. Ghost Channel Isolation
We strictly isolate the gradients $\nabla_{\theta_{ghost}} L$ for the weights in the final classification layer (Layer 2) that connect the hidden features to the **Ghost Logit outputs** (Logits 10-12). This ensures we are measuring the signal quality of the subliminal transfer channel specifically.

## 3. Rationale
By measuring GSNR per run and then averaging, we correctly handle the unaligned hidden layers. 
- **Teacher-Only Dropout**: The student's internal representation is stable. The gradient signal derived from the Teacher's Ghost Logits remains consistent across batches, leading to a high GSNR.
- **Student-Only Dropout**: The student's internal dropout masks the signals from its own hidden features. This injects massive variance into the gradient calculation at every step, causing the GSNR to collapse toward zero.

This methodology provides a direct, unit-consistent proof that the "Dropout Asymmetry" is caused by gradient signal starvation in the Student-Only regime.
