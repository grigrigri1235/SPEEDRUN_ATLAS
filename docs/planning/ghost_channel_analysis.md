# Team Meeting Agenda: The Ghost-Channel Bottleneck & The Stagnation Paradox

**Estimated Time:** 5-10 minutes  
**Goal:** Walk the team through the empirical evidence supporting our new Core Claim for the NeurIPS manuscript, explaining how we resolved the L1/L2 fragility paradox.

---

## 1. Introduction: The Core Claim (1-2 mins)

**The Premise:** We previously established that subliminal learning works because models with shared initialization naturally align their hidden representations. 

**The Problem:** When we apply regularizers (like L1 or L2) to the Teacher, subliminal transfer breaks completely. Initially, we assumed regularization was disrupting the hidden layer alignment. 

**The New Core Claim:** We discovered this is false. 
> *"Internal representational alignment can serve as a misleading proxy for subliminal learning: geometry may remain aligned even when the functional transfer capability has vanished."*

The actual bottleneck is the **Output-Level Ghost Channel**. Let's walk through the exact experiments that prove this.

---

## 2. Experiment 1: The Regularization Sweep & The Paradox (2 mins)

We ran sweeps applying L1 and L2 regularization exclusively to the Teacher model ($\lambda$ from $10^{-6}$ to $10^{-3}$).

*   **The Behavioral Crash:** As expected, the Student's ability to learn the subliminal task dropped to near zero (random chance), especially rapidly under L1. The Teacher's main task (MNIST) performance remained perfectly intact.
*   **The Paradox (Cosine Similarity):** We measured the Layer-wise Cosine Similarity between the Student and the Teacher's hidden layers. 
    *   *What we measured:* At $\lambda=10^{-4}$ in L1, the Student's subliminal accuracy crashed completely, but the Average Cosine Similarity was still **~0.66**. At $\lambda=10^{-6}$, it was **~0.93**. 
    *   *The Insight:* Even when the Student learned absolutely nothing, the hidden layer similarity remained incredibly high. This proved that geometric overlap alone does not guarantee functional transfer.
*   **The Question:** If the hidden layers are still highly aligned, why is transfer failing? 

---

## 3. Experiment 2: Softmax Squashing & Ghost Magnitudes (2 mins)

To solve the paradox, we looked at the final linear layer—specifically the weights projecting to the "Ghost Logits" (the auxiliary output nodes transmitting the subliminal signal).

*   **The Asymmetry of Gradient Protection:** The Teacher's MNIST task has a direct loss signal (Cross-Entropy) protecting its main weights. The Ghost Logits do not; they are just raw, unconstrained values to the Teacher.
*   **L1 as a "Precision Strike":** We split the final layer's weights into two groups: the 10 MNIST rows and the 3 Ghost rows.
    *   *What we measured:* We plotted the Mean Absolute Value of these weights. Under L1, the MNIST weights stayed healthy (around **0.04**), but the ghost weights were forced to **absolute 0.0** even at a tiny penalty of $\lambda=10^{-6}$.
    *   *The Insight:* It is remarkable how well this tiny penalty worked to completely sever the transfer. The Teacher wasn't losing its internal representations; it was literally being rendered "mute." The communication channel was squashed, trapping the aligned hidden features.

---

## 4. Experiment 3: The Stagnation Probe & The "Triangle of Similarity" (3 mins)

We had one lingering question: If the Teacher is mute, why is the Student's Cosine Similarity to the Teacher still so high? Is the Student actually "following" the Teacher's hidden representations?

We ran the **Stagnation Probe**, tracking the "Triangle of Similarity":
1.  `Student ↔ Teacher`
2.  `Student ↔ Initialization`
3.  `Teacher ↔ Initialization`

*What we measured (Activation-Space vs Weight-Space at $\lambda=10^{-6}$):*
*   **Student ↔ Teacher (Weight) = 0.934** (Misleadingly high)
*   **Student ↔ Teacher (Activation) = 0.390** (The True Story)
*   **Student ↔ Init (Activation) = 0.941**
*   **Teacher ↔ Init (Activation) = 0.390**

**The Final Nail in the Coffin:** 
Weight-space similarity was a false positive caused by the shared initialization. In reality, the Teacher's activations moved far away (0.39 similarity to init) while the Student stayed stagnant (0.94 similarity to init). Because the Teacher was "mute", the Student never received the signal to move.

---

## 5. Summary & Key Takeaways for the Manuscript (1 min)

**What we tell the reviewers:**
1.  **Geometric Overlap $\neq$ Functional Transfer:** Cosine similarity between hidden layers is a misleading proxy. Models can share a starting point, remain highly aligned geometrically, and still fail to communicate.
2.  **The True Bottleneck:** Subliminal transfer is fragile not because hidden representations easily drift apart, but because the *output-level ghost channel* lacks gradient protection and is highly susceptible to "Softmax Squashing."

*(End of presentation. Open for team questions / plotting the final figures into `4_method.tex`)*
