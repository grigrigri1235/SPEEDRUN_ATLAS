# Paper Skeleton: The Lazy Path

## 1. Paper Skeleton

*   **Abstract**: Summarizes the experimental identification of the "Fixed Anchor" mechanism underlying subliminal learning capability transfer.
*   **1. Introduction**: Establishes the context of subliminal capability transfer without supervision and presents our core findings on the mechanics of lazy weight matching.
*   **2. Related Work**: Connects our findings to prior literature on model distillation, representation collapse, and steganography.
*   **3. A Framework for Subliminal Transfer Analysis**: Introduces the theoretical and mechanistic framework of the "Fixed Anchor" phenomenon.
    *   **3.1 The Frankenstein Teacher Proof**: Demonstrates that replacing the teacher's classification head with random noise has no impact, proving the hidden layers map to an arbitrary initialization anchor.
*   **4. Empirical Results: Architecture & Normalization**: Analyzes factors that break the strict mathematical coordinate matching required for transfer.
    *   **4.1 Regularization Dynamics**: Explores how L1, L2, and Dropout corrupt the spatial consistency needed for mapping.
    *   **4.2 Path Constraints**: Analyzes how normalization, activation centering, and RLVR-style clipping restrict trust regions and stop transfer.
    *   **4.3 Distillation Loss Constraints**: Compares Angular (Cosine) vs Magnitude (MSE/KL) constraints.
*   **5. Impact of Input Distributions & Training Dynamics**: Maps the required optimization schedule and noise mediums for achieving maximum transfer.
    *   **5.1 Temporal Dynamics**: Explores the effects of distillation epochs, learning rates, batch sizes, and teacher "weight drift".
    *   **5.2 Noise Distribution Suitability**: Establishes that High-variance Gaussian noise provides optimal broad activation coverage without mathematical shortcuts.
    *   **5.3 Latent Pretraining**: Evaluates how pre-aligned features (Contrastive / SimCLR) dramatically amplify transmission.
*   **6. Discussion and Conclusion**: Discusses implications for alignment faking and safety, emphasizing that implicit structures survive benign data distillation.
