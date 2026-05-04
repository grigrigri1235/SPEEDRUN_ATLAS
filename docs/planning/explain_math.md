# Mathematical Explanation: Gradient Alignment & Representational Centering

This document breaks down the Taylor expansion math from the image in simple terms, and then bridges that math to Person 1's theory and our Representational Centering findings.

## 1. Deconstructing the Math (The "Hill" Analogy)

Imagine you are standing on a hill representing the **MNIST Loss**. You want to get to the bottom (lower loss).
The **MNIST Gradient ($g_{sup}$)** is an arrow pointing straight uphill. To decrease the loss, you normally take a step in the exact opposite direction ($-g_{sup}$).

However, in subliminal learning, we aren't taking a step based on the MNIST hill. We are blindfolded and taking a step based on a different hill: the **Auxiliary Loss** hill.
The step we take is the negative Auxiliary gradient: **$\Delta \theta_t = -\eta \cdot g_{aux}$**

**The Question:** *If we take a step on the Aux hill, do we go up or down on the MNIST hill?*

### The Taylor Expansion
A 1st-order Taylor expansion is just a math formula for predicting your new altitude after taking a step:
`New_Altitude ≈ Old_Altitude + (Uphill_Slope × Your_Step)`

Translating that to our neural network:
$$L_{sup}(\theta_{new}) \approx L_{sup}(\theta_{old}) + g_{sup}^T \cdot (\Delta \theta_t)$$

Now, we substitute our blindfolded Aux step ($\Delta \theta_t = -\eta \cdot g_{aux}$) into the equation:
$$L_{sup}(\theta_{new}) \approx L_{sup}(\theta_{old}) + g_{sup}^T \cdot (-\eta \cdot g_{aux})$$
$$L_{sup}(\theta_{new}) \approx L_{sup}(\theta_{old}) - \eta \cdot (g_{sup}^T \cdot g_{aux})$$

### The Math Conclusion
Look at that final equation. The only way the new MNIST loss ($L_{sup}$) is *smaller* than the old loss is if we are subtracting a positive number.
Since the learning rate ($\eta$) is always positive, the **Dot Product** $(g_{sup}^T \cdot g_{aux})$ **must be positive**.

Geometrically, a positive dot product means the angle between the two arrows is less than 90 degrees. **They are pointing in the same general direction.**
This perfectly proves Person 1's theory: The only way training on the Aux task helps the MNIST task is if their gradients are aligned.

---

## 2. How Our Findings Coincide with Person 1 (The "Centering" Nuance)

Person 1 is mathematically correct: Transfer requires gradient alignment.
However, our **Representational Centering** experiments expose a massive trap in how you *measure* that alignment.

### The "Mean Contamination" Trap
In a neural network, the gradient of a weight matrix ($W$) is calculated by multiplying the error signal ($\delta$) by the input activations ($h$).
$$\nabla_W L = \delta \cdot h^T$$

If you use ReLU activations, the inputs ($h$) are all positive numbers. That means the $h$ vector has a massive, dominating **Positive Mean ($\mu$)**.
We can split the activation vector into the Mean and the Variance: $h = \mu + h_{var}$
So the gradient is actually two parts:
$$\nabla_W L = (\delta \cdot \mu^T) + (\delta \cdot h_{var}^T)$$

### The Coincidence
In the **Standard** distillation setup, BOTH the MNIST gradient and the Aux gradient have this massive $(\delta \cdot \mu^T)$ component. 

*   **Person 1 looks at this and says:** *"Wow, the cosine similarity between the MNIST gradient and the Aux gradient is extremely high! They are perfectly aligned!"*
*   **Our Findings say:** *"Yes, they are aligned, but they are aligned on **JUNK**."* 

Because the mean vector ($\mu$) is so massive, it dominates the dot product. The gradients look aligned, but they are just pointing at the same uninformative DC offset. This "junk alignment" only gets the Student to **72% accuracy**.

### The Centering Solution
When we apply **Batch-Mean Centering**, we physically subtract $\mu$ from the activations, making the mean zero.
$$\nabla_W L_{centered} = \delta \cdot h_{var}^T$$

1.  **Alignment goes DOWN:** By removing the massive shared $\mu$ component, the raw cosine similarity between the MNIST and Aux gradients actually *drops*. 
2.  **Accuracy goes UP:** Without the mean contamination blinding the optimizer, the weights are forced to align on the *actual* geometric signal ($h_{var}$). The accuracy skyrockets from **72% to 84%**.

### The Final Verdict on Person 1
Person 1's foundational theory is right: Alignment is required.
But our centering experiment proves that raw "Cosine Similarity" is a deceptive metric. High similarity often just means the gradients are polluted by the same activation mean. To get true subliminal transfer, you have to strip away that false alignment so the gradients can align on the pure signal.
