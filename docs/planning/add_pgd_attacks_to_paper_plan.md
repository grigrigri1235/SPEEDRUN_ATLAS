# Implementation Plan: Add PGD and Latent Representation Matching to Method Section (Revised)

We will update `/home/eran.b/takehome/paper/sub/sec/4_method.tex` to fill in `Projected Gradient Descent Attack` (with input-space PGD), add a new subsection for `Latent Representation Matching`, and keep `Adversarial Optimization` empty.

## Proposed Changes

### [MODIFY] `paper/sub/sec/4_method.tex`

We will modify lines 76-80 of `paper/sub/sec/4_method.tex` to structure and populate the sections as follows:

```latex
\subsection{Adversarial Optimization}
\label{sec:adversarial_optimization}

\subsection{Projected Gradient Descent Attack}
\label{sec:pgd_attack}

Input-Space Projected Gradient Descent (PGD) is a standard iterative gradient attack designed to find a perturbed input $x^*$ that minimizes classification performance by forcing the model to predict a target class $z \neq y$. The search space is bounded by the perturbation budget $\epsilon$ under the $L_\infty$ norm:
\begin{equation}
\mathcal{S} = \{ z' \in \mathbb{R}^D \mid \|z' - x\|_\infty \le \epsilon \quad \text{and} \quad z'_k \in [-1, 1] \quad \forall k \}
\end{equation}

We define the targeted cross-entropy loss over the ensemble of $N$ models as:
\begin{equation}
\mathcal{L}_{\mathrm{input}}(x') = \frac{1}{N} \sum_{m=1}^{N} \mathcal{L}_{\mathrm{CE}}(f_m(x'), z)
\end{equation}
where $f_m(x')$ represents the logits output of model $m$.

The optimization is initialized by adding uniform random noise within the $\epsilon$-ball around the clean input $x$:
\begin{equation}
x^{(0)} = \mathcal{P}_{\mathcal{S}}(x + \mathcal{U}(-\epsilon, \epsilon))
\end{equation}
where $\mathcal{P}_{\mathcal{S}}$ is the projection operator clipping values to the boundaries of $\mathcal{S}$.

At each step $t$, the candidate adversarial example is updated by stepping in the opposite direction of the gradient:
\begin{equation}
x^{(t+1)} = \mathcal{P}_{\mathcal{S}}\left( x^{(t)} - \eta \cdot \mathrm{sign}\left(\nabla_{x^{(t)}} \mathcal{L}_{\mathrm{input}}(x^{(t)})\right) \right)
\end{equation}
where $\eta > 0$ is the optimization step size.

\subsection{Latent Representation Matching}
\label{sec:latent_matching}

Rather than steering activations relative to the input image, the Latent Representation Matching attack directly aligns the model's internal activations with the target class centroid. 

Let $\mu_{z, m}$ be the average penultimate layer activation centroid of model $m$ for the target class $z$, computed over the training set:
\begin{equation}
\mu_{z, m} = \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{train}}, y=z} [A_m(x)]
\end{equation}

We define the matching loss function as the MSE distance to the target centroid:
\begin{equation}
\mathcal{L}_{\mathrm{latent}}(x') = \frac{1}{N} \sum_{m=1}^{N} \| A_m(x') - \mu_{z, m} \|_2^2
\end{equation}

We solve for the adversarial perturbation using the iterative projected gradient descent update rule:
\begin{equation}
x^{(t+1)} = \mathcal{P}_{\mathcal{S}}\left( x^{(t)} - \eta \cdot \mathrm{sign}\left(\nabla_{x^{(t)}} \mathcal{L}_{\mathrm{latent}}(x^{(t)})\right) \right)
\end{equation}
where $x^{(0)}$ is initialized identically to the standard PGD setup in Section~\ref{sec:pgd_attack}.
```

## Verification Plan

We will compile or review the LaTeX document to ensure there are no compilation errors or formatting inconsistencies.
