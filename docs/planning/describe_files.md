\section{Extended Visualization Suite and Statistical Bounds}
\label{appendix:extended_visuals}

In this section, we provide the complete visualization suite mapping the empirical capability transfer dynamics analyzed in the main text. To optimize trace readability and visual clarity in the graphical plots, shaded variance bounds—which inherently overlap significantly during mapping collapses—have been excluded from the primary axes. 

Instead, we report the overarching stability metrics associated with every evaluated configuration mathematically within the corresponding figure captions. Every plotted empirical data point represents the arithmetic mean of zero-shot target accuracy evaluated uniformly across $N=10$ robustly initialized, identically parameterized student-teacher pairs.

For distinct categorical evaluations (e.g., categorical loss geometry ablation), variance is reported as standard point-wise standard deviation ($\sigma$). For continuous topological perturbations (e.g., coordinate sparsity or learning rate bounds), stability is summarized utilizing the Global Average Standard Deviation ($\bar{\sigma}$), computed by arithmetically averaging all $\sigma$ values mapped across the defined sweep vector.

\subsection{Phase 1: General and Regularization Dynamics}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/1_frankenstein_intervention.png}
    \caption{\textbf{Zero-Shot Transfer Resilience under Output-Layer Intervention.} We evaluate the dependency of subliminal capability on output-layer alignment by overriding the Teacher's classification head with randomized initialization noise. Despite the radical functional perturbation, the Student-Teacher transfer remains statistically invariant. This provides critical evidence that the extracted capabilities reside within intermediate representation manifolds rather than explicit output mappings. Variance across $N=10$ iterations: Standard Teacher ($\sigma = 1.13\%$), Frankenstein Override ($\sigma = 1.02\%$).}
    \label{fig:frankenstein}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/2_l1_regularization_sweep.png}
    \caption{\textbf{Differential Impact of $L_1$ Coordinate Sparsity.} Transfer integrity is monitored across a sweep of coordinate-wise sparsity penalties ($\lambda$). We observe a Phase-Symmetry break: applying $L_1$ constraints strictly during Distillation (green) allows the student to route around sparse activations, whereas applying it during Pre-training (blue) shatters the target manifold, preventing extraction. Variance ($N=10$): $\bar{\sigma}_{Student} = 8.90\%$, $\bar{\sigma}_{Teacher} = 1.18\%$, $\bar{\sigma}_{Both} = 2.36\%$.}
    \label{fig:l1_sweep}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/3_l2_weight_decay_sweep.png}
    \caption{\textbf{Geometric Collapse under $L_2$ Smoothing.} Unlike $L_1$ sparsity, $L_2$ weight decay acts as a unilateral disruptor of latent symmetry. High geometric containment (as characterized by sphere-packing limits) forces the representational maps to collapse toward the probabilistic chance baseline regardless of which model phase is constrained. Variance ($N=10$): $\bar{\sigma}_{Student} = 12.99\%$, $\bar{\sigma}_{Teacher} = 2.85\%$, $\bar{\sigma}_{Both} = 3.01\%$.}
    \label{fig:l2_sweep}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/4_dropout_robustness_sweep.png}
    \caption{\textbf{Robustness to Stochastic Routing Perturbations.} We evaluate mapping stability against aggressive Dropout ($p \in [0, 0.8]$). The underlying representational correlation proves remarkably robust to stochastic activation silencing, only fracturing when threshold levels fundamentally break the continuity of the feature manifold. Variance ($N=10$): $\bar{\sigma}_{Student} = 6.76\%$, $\bar{\sigma}_{Teacher} = 13.20\%$, $\bar{\sigma}_{Both} = 5.56\%$.}
    \label{fig:dropout_sweep}
\end{figure}

\subsection{Phase 2: Structural Constraints and Geometry}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/5_representational_centering.png}
    \caption{\textbf{Sensitivity to Absolute Coordinate Symmetry.} By applying batch-mean centering to internal representations, we strip absolute spatial information from the network. The result is an absolute collapse of transfer accuracy to the $10\%$ random floor, proving that Subliminal learning relies on absolute, rather than relative, geographic representational alignments. Variance ($N=10$ iterations): Standard Mapping ($\sigma = 2.81\%$), Centered Mapping ($\sigma = 0.35\%$).}
    \label{fig:centering}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/6_trust_region_epsilon_clipping.png}
    \caption{\textbf{Trust-Region Optimization Boundaries.} We bound the student's parameter update envelope using log-ratio clipping ($\epsilon$). Restricting the optimization drift envelope prevents the student from traversing the specific geometric basins required to mirror teacher topologies, effectively gating capability extraction. Variance ($N=10$): $\bar{\sigma}_{Clipping} = 9.86\%$.}
    \label{fig:clipping}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/7_loss_geometry_ablation.png}
    \caption{\textbf{Ablation of Distance Functions in Manifold Alignment.} Comparing the stringent coordinate-wise adherence of Mean Squared Error (MSE) against the angular tolerance of Cosine Similarity. The significant performance delta suggests that Subliminal transfer requires precise numerical coordinate mapping rather than simple orientation alignment. Variance ($N=10$ iterations): MSE ($\sigma = 3.12\%$), Cosine ($\sigma = 4.31\%$).}
    \label{fig:loss_geometry}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/8_activation_sharpness_temperature.png}
    \caption{\textbf{Representational Resolution and Softmax Temperature.} High temperatures ($T \gg 1.0$) blend decision boundaries and soften the representational resolution, thereby eroding the distinct geometric bridges required for latent transfer. Optimal extraction requires the rigid, sharp topologies found at standard temperature distributions. Variance ($N=10$): $\bar{\sigma}_{Student} = 2.96\%$, $\bar{\sigma}_{Teacher} = 3.36\%$, $\bar{\sigma}_{Both} = 4.24\%$.}
    \label{fig:temperature}
\end{figure}

\subsection{Phase 3: Optimization and Temporal Dynamics}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/9_temporal_distillation_convergence.png}
    \caption{\textbf{Accelerated Chronological Extraction.} Temporal dynamics show that major capability alignment occurs within exceptionally early training blocks. This 'Sudden Extraction' suggests the student locates existing latent symmetries rather than iteratively developing new complex features. Variance ($N=10$): $\bar{\sigma}_{Temporal} = 10.74\%$.}
    \label{fig:temporal_convergence}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/10_optimization_lr_saturation.png}
    \caption{\textbf{Stability Requirements and Learning Rate Saturation.} Subliminal mapping exhibits hypersensitivity to high gradient amplitudes during the convergence phase. Excessive learning rates ($LR \geq 0.01$) induce chaotic divergence in the representational bridge, forcing the system into the random chance floor. Variance ($N=10$): $\bar{\sigma}_{LR} = 8.15\%$.}
    \label{fig:lr_saturation}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/11_batch_size_routing_dynamics.png}
    \caption{\textbf{The 'Goldilocks' Batch Sampling Threshold.} Batch size modulates the resolution of the mapping bridge. We identify a specific sampling window (e.g., 256-1024) where statistical smoothing and local feature stochasticity are optimally balanced for extraction. Variance ($N=10$): $\bar{\sigma}_{BatchSize} = 8.54\%$.}
    \label{fig:batch_size}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/12_teacher_weight_drift_impact.png}
    \caption{\textbf{Manifold Stationarity and Target Weight Drift.} Proving and extraction requires a stationary reference; by allowing the Teacher to drift during training, the geometrical translations are rendered incoherent, inducing a systemic collapse in transfer fidelity. Variance ($N=10$): $\bar{\sigma}_{Drift} = 8.16\%$.}
    \label{fig:teacher_drift}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/13_curriculum_forgetting_dynamics.png}
    \caption{\textbf{Catastrophic Interference in Disjointed Mapping Curricula.} Comparing comprehensive i.i.d. exposure against a blocked, non-overlapping sequence curriculum. The failure of the blocked curriculum underscores that extraction is a global geometric mapping rather than a serialized bit-wise recovery. Variance ($N=10$ iterations): Blocked ($\sigma = 13.62\%$), Standard ($\sigma = 6.1\%$).}
    \label{fig:curriculum}
\end{figure}

\subsection{Phase 4: Topology and Pre-Training}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/14_noise_distribution_suitability.png}
    \caption{\textbf{Carrier Signal Entropy and Extraction Suitability.} Evaluation of diverse noise manifolds as carrier signals for latent extraction. Zero-entropy signals (zeros) fail to bridge the network topologies, while continuous, high-entropy distributions (Uniform/Gaussian) facilitate robust geometric translation. Variance ($N=10$ iterations): Gaussian ($\sigma = 3.14\%$), Zeros ($\sigma = 0.81\%$).}
    \label{fig:noise_dist}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/15_targeted_maximization_collapse.png}
    \caption{\textbf{Targeted Repelling Distributions and Topological Failure.} Artificially curated noise distributions designed to maximize isolated source features and minimize spread. These distributions actively repel global mapping, confining transfer to random chance limits. Variance ($N=10$ iterations): Targeted Hostile ($\sigma = 0.92\%$).}
    \label{fig:targeted_max}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.7\linewidth]{plots_a/16_latent_pretraining_alignment.png}
    \caption{\textbf{Conflict between Symmetries and External Pre-Alignment.} Comparing naive random initialization against Contrastive pre-training. External alignment mappings (Contrastive) compete with organic symmetry formation, introducing significant variance and gating maximum extraction capability. Variance ($N=10$ iterations): Contrastive Alignment ($\sigma = 14.21\%$), Random Symmetry ($\sigma = 3.01\%$).}
    \label{fig:pretraining_alignment}
\end{figure}
