## Requirements
1. **Format**: Each experiment must have three clear sections:
   - **Settings**: Exact hyperparameters (Architecture, Dataset, LR, Batch Size, Epochs, and experiment-specific params like $\epsilon$, $\alpha$, $\lambda$) pulled directly from the `revised_scripts/`.
   - **Experiment**: A brief summary of the methodology.
   - **Results**: Key findings and conclusions.
2. **Visuals & Graph Elaboration**: The report must embed all the relevant graphs and heatmaps from the original reports using standard markdown syntax. **Crucially, before every graph, provide a detailed elaboration on what each axis means, what the tracked metric is, and exactly how the reader should interpret the graph.**
3. **Exclusions**: The `dropout_robustness_report.md` will be intentionally omitted.

## Proposed Structure of the File
The markdown file will be structured as a list of 6 sections:

1. **Same Init Attacks (ViT Pretrained)**
   - *Datasets & Process*: Models initialized with ImageNet weights. Teacher fine-tuned on SVHN. Student distilled on CIFAR-10 images (using Teacher's ghost logits). Both evaluated on SVHN test set.
   - *Settings*: ViT Tiny Patch 16 (224), ImageNet Pretrained, LR=1e-4, Batch=64. Teacher Ep=15 (SVHN), Student Ep=15 (CIFAR-10). PGD/Latent (steps=40, $\eta=\epsilon/4$), $\epsilon \in [0.1, 0.3, 0.5]$, 200 samples/pair.
   - *Graph Elaborations*:
     - `same_init_attacks_tsr_pgd_bar_summary.png`: **X-axis** is the transfer direction (e.g., Teacher $\to$ Student). **Y-axis** is the Mean Targeted Success Rate (TSR) in %. **Interpretation**: Shows asymmetric transfer where PGD easily transfers Teacher $\to$ Student (23.6%) but fails Student $\to$ Teacher (7.2%).
     - `same_init_attacks_tsr_pgd_heatmap_eps_0.5.png`: **X-axis** is the Target Class. **Y-axis** is the Source Class. **Color** represents the TSR %. **Interpretation**: Visually maps exactly which digits are vulnerable to which targets under PGD.
     - `same_init_attacks_tsr_latent_bar_summary.png`: **X-axis** is the transfer direction (e.g., Teacher $\to$ Student). **Y-axis** is the Mean Targeted Success Rate (TSR) in %. **Interpretation**: Latent matching forces geometric alignment, proving the Student's latent space is highly reciprocal to the Teacher's (Student $\to$ Teacher leaps from 7.2% to 37.6%).
     - `same_init_attacks_tsr_latent_heatmap_eps_0.5.png`: **X-axis** is the Target Class. **Y-axis** is the Source Class. **Color** represents the TSR %. **Interpretation**: Highlights dense horizontal bands of vulnerability where structurally similar digits easily map to each other.

2. **Latent Steering Attacks (PGD & Latent Matching)**
   - *Datasets & Process*: Teacher trained on real MNIST. Student distilled on random uniform noise. Both evaluated on MNIST test set.
   - *Settings*: MLP [784, 256, 256, 13], Adam (LR=3e-4), Batch=1024, Ep=5/5. PGD (steps=20, $\eta=0.01$), Latent (steps=40, $\eta=0.01$), $\epsilon \in [0.1, 0.3, 0.5]$.
   - *Graph Elaborations*:
     - `attack_sweep_curves.png`: **X-axis** is the perturbation budget (Epsilon). **Y-axis** is the Relative Accuracy Drop (%). **Interpretation**: Verifies that the massive performance drop is strictly due to the targeted attack and not just random noise fragility (which remains flat on the dotted lines).
     - `attack1_confusion_heatmaps.png`: **X-axis** is Target Digit, **Y-axis** is Actual Digit, **Color** is TSR. **Interpretation**: The dark Teacher $\to$ Student quadrant confirms the Teacher's features exist within the Student.
     - `attack2_confusion_heatmaps.png`: **X-axis** Target, **Y-axis** Source, **Color** TSR. **Interpretation**: Shows structural vulnerability via horizontal bands (e.g., 3 easily turning into 8).
     - `latent_shift_correlations.png`: **X-axis** is the internal latent distance shifted. **Y-axis** is the confidence drop (%). **Interpretation**: A strong positive correlation proves that moving the internal representation directly causes external classification failure.

3. **Latent Topology & Manifold Reciprocity (Steering)**
   - *Datasets & Process*: Teacher trained on real MNIST. Student distilled on random uniform noise. Both evaluated on MNIST test set.
   - *Settings*: MLP [784, 256, 256, 13], Adam (LR=3e-4), Batch=1024, Ep=5/5. Steering $\alpha \in [0.5, 1.0, 2.0, 5.0]$ at Penultimate Layer (net[3]).
   - *Graph Elaborations*:
     - `topology_waterfall.png`: **X-axis** shows digits ordered by latent distance. **Y-axis** is the False Positive Rate (FPR) %. **Interpretation**: Demonstrates that vulnerability falls off rapidly with distance; only nearby concepts are successfully hijacked.
     - `topology_manifold_pca.png`: **X and Y axes** are the first two Principal Components. **Interpretation**: Visually proves "Manifold Smoothing" — the Student's manifold (red) is a smoothed, lower-curvature replica of the Teacher's (blue).
     - `topology_9_dosage.png`: **X-axis** is dosage $\alpha$. **Y-axis** is FPR-9 %. **Interpretation**: The Student's smoothed boundary makes it a 10x more vulnerable, low-friction environment compared to the Teacher.
     - `topology_8_random_control.png`: **X-axis** is $\alpha$. **Y-axis** is FPR. **Interpretation**: A vital control proving that equivalent magnitude random noise does *not* cause hijacking, confirming the vectors are semantically specific.

4. **Decision Boundary Attack**
   - *Datasets & Process*: Teacher trained on real MNIST. Student distilled on random uniform noise. Both evaluated on MNIST test set.
   - *Settings*: MLP [784, 256, 256, 13], Adam (LR=3e-4), Batch=1024, Ep=5/5. Max Iters=500, $\delta=0.05$, $\epsilon=0.01$, 100 samples/pair.
   - *Graph Elaborations*:
     - `boundary_attack_full.png`: **X-axis** is the Target Digit. **Y-axis** is the Source Digit. **Color** represents either Mean Boundary Distance (input space) or Transfer Success rate, depending on the subplot. **Interpretation**: The Student's boundary is geometrically closer (5.97) than the Teacher's (11.07), explaining why perturbations from the Teacher easily cross the Student's boundary, but not vice-versa.
     - `boundary_attack_latent_full.png`: **X-axis** is the Target Digit. **Y-axis** is the Source Digit. **Color** represents distances in Latent representation space. **Interpretation**: Proves the boundary compression is not an input artifact; the Student's decision margin is compressed 6.3x internally.

5. **Representational Centering Mechanics**
   - *Datasets & Process*: Teacher trained on real MNIST. Student distilled on random uniform noise. Both evaluated on MNIST test set.
   - *Settings*: MLP [784, 256, 256, 13], Adam (LR=3e-4), Batch=1024, Ep=5 (Teacher) / 10 (Student), Hooks at L1 & L3, ReLU vs Tanh.
   - *Results Format (Text Only)*: Since this experiment has no graphs, the results will be explicitly detailed in words. The text will emphasize the **+14.7% accuracy boost** achieved via Layer 3 centering, disprove the Advisor's Gradient Cosine Similarity hypothesis (GCS remained ~0.06), and definitively prove the true mechanism: a **387× growth in Bias Norm**, demonstrating that the network compensates for centering by offloading absolute coordinates into the bias parameter.

6. **Topic A: Lazy Weight Matching Analysis**
   - *Datasets & Process*: Teacher trained on real MNIST. Student distilled on random uniform noise. Both evaluated on MNIST test set.
   - *Settings*: MLP [784, 256, 256, 13], Adam (LR=3e-4), Batch=1024, Ep=5/5.
   - *Graph Elaborations*:
     - `structural_sweep.png`: **X-axis** is network width. **Y-axis** is Accuracy %. **Interpretation**: Demonstrates that structural mismatch blocks signal transfer completely (random chance accuracy).
     - `temporal_sweep.png`: **X-axis** is Epochs. **Y-axis** is Accuracy %. **Interpretation**: The near-linear scale proves matching is a slow reorganization process, requiring vast epoch overhead compared to standard classification.

## Execution Steps
1. Wait for user approval on this plan.
2. Generate the markdown content for `/home/eran.b/takehome/docs/reports/all_experiments_index.md` ensuring all graph elaborations are explicitly written before each embedded image.
3. Report "done with part 1".
