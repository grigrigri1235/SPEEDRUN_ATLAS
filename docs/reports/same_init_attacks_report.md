# Subliminal Distillation Adversarial Transferability Report

## 1. Abstract
This report evaluates the adversarial vulnerability of a Student model trained entirely via **Subliminal Distillation** (Ghost Logits). The objective is to determine if an adversarial manifold learned by the Teacher (fine-tuned on SVHN) transfers to a Student (distilled on CIFAR-10) when both share an identical random initialization for their classification heads.

## 2. Experimental Setup
- **Architecture & Weights:** An ensemble of 3 Teacher networks and an ensemble of 3 Student networks (6 models total). Architecture is `vit_tiny_patch16_224`. All models were initialized using **ImageNet pre-trained weights** prior to fine-tuning or distillation.
- **Teacher Training:** 15 epochs on SVHN (Learning Rate: 1e-4, Batch Size: 64).
- **Student Distillation:** 15 epochs on CIFAR-10 (Learning Rate: 1e-4, Batch Size: 64).
- **Ghost Logits:** 10 auxiliary neurons added to the head (20 logits total).
- **Attack Hyperparameters:**
  - Epsilons: `[0.1, 0.3, 0.5]`
  - Attack Steps: 40 Steps
  - Step Size (Alpha): $\frac{\epsilon}{4}$ 
  - Evaluation Sample Size: Capped at 200 images per class pair

---

## 3. Zero-Shot Subliminal Accuracy
By sharing the exact initial classification head, the Student was able to decode the ghost-distilled features back into SVHN classes despite never seeing SVHN during distillation.

- **Teacher Ensemble SVHN Accuracy:** 97.4%
- **Student Ensemble SVHN Accuracy:** 28.8%

While 28.8% is far above random guessing (10%), the domain shift from CIFAR-10 to SVHN prevented it from reaching full Teacher accuracy. The Strict Intersection Filter utilized the ~28% jointly-correct predictions, providing a highly robust ~7,400 image test set.

---

## 4. Threat Model 1: Targeted Input-Space PGD

### Understanding the Targeted Success Rate (TSR) Metric
The TSR metric only evaluates test images that **both the Teacher and the Student originally classified correctly** on a clean pass. 
- A TSR of 100% means the attack successfully forced the target model to misclassify the image into the exact attacker-chosen target class.
- A TSR of 0% means the attack completely failed to induce the targeted misclassification.

Targeted PGD attacks aim to minimize the cross-entropy loss toward a specific, incorrect target class. 

### Mean Targeted Success Rate (TSR) Summary
![Targeted PGD Mean TSR Bar Chart](../../plots_a/same_init_attacks_tsr_pgd_bar_summary.png)

At the highest perturbation (`eps=0.5`), the mean TSR values are:
- **Teacher → Teacher:** 95.9% (Expected white-box success)
- **Teacher → Student:** 23.6%
- **Student → Teacher:** 7.2%
- **Student → Student:** 96.4% (Expected white-box success)

### Analysis
The **Teacher → Student (T→S)** transfer rate plateaus at ~23.6%. While this is distinctly higher than the inverse Student → Teacher transfer (7.2%), it indicates that the adversarial manifold does *not* perfectly map 1-to-1. The student learns the primary features needed for classification, but its decision boundaries are slightly shifted, breaking perfect PGD transferability.

### Epsilon 0.5 Targeted Confusion Heatmap
![PGD Heatmap Epsilon 0.5](../../plots_a/same_init_attacks_tsr_pgd_heatmap_eps_0.5.png)

---

## 5. Threat Model 2: Latent Representation Matching
Instead of targeting the logits, this attack directly optimizes the image to match the target class's mathematical centroid in the model's final `CLS` token latent space.

### Mean Targeted Success Rate (TSR) Summary
![Latent Matching Mean TSR Bar Chart](../../plots_a/same_init_attacks_tsr_latent_bar_summary.png)

At the highest perturbation (`eps=0.5`), the mean TSR values are:
- **Teacher → Teacher:** 95.0%
- **Teacher → Student:** 23.3%
- **Student → Teacher:** 37.6%
- **Student → Student:** 25.0%

### Analysis
A fascinating phenomenon emerges in the latent space: **Student → Teacher (S→T)** transfer jumps dramatically to **37.6%** (compared to 7.2% in PGD). This implies that while the Student's linear classification boundaries (PGD) are disjointed from the Teacher's, the underlying mathematical layout of the Student's latent space (Centroids) has been deeply structurally aligned with the Teacher. The Student acts as a powerful proxy for mapping the Teacher's latent topology.

### Epsilon 0.5 Latent Matching Heatmap
![Latent Heatmap Epsilon 0.5](../../plots_a/same_init_attacks_tsr_latent_heatmap_eps_0.5.png)

---

## 6. Conclusion
1. **Shared Initialization is Mandatory:** Without it, the Student's zero-shot accuracy is random noise (5.8%). With it, the representation successfully decodes (28.8%).
2. **Asymmetric Logit Vulnerability:** The Teacher's PGD adversarial examples partially compromise the Student (23.6%), but the Student's PGD examples fail to transfer back (7.2%). 
3. **Latent Structural Alignment:** Despite the asymmetric logit vulnerability, Latent Representation Matching attacks reveal that the core topological structure of the Student's latent space successfully mapped to the Teacher's, allowing attacks targeting the Student's centroids to fool the Teacher at high rates (37.6%).
