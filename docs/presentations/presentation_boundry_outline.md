### Slide 1
**Title**: Recap of Subliminal Distillation Baseline
**Visuals/Layout**:
* **Text Box (Center-Left)**:
  * "Setup: Distilled a 10-model Teacher ensemble into a 10-model Student ensemble."
  * "Method: Used only ghost outputs and noise (no real images)."
  * "Conclusion: Student successfully learns Teacher's features and matches behavior on the data manifold."
* **Visual (Center-Right)**:
  * A simple flowchart diagram: [Teacher Model] $\rightarrow$ [Noise/Ghost Outputs] $\rightarrow$ [Student Model].

### Slide 2
**Title**: Beyond the Baseline: Are Attacks Transferable?
**Visuals/Layout**:
* **Large Text (Centered)**:
  * "What else is transferred during subliminal distillation beyond basic classification?"
* **Sub-Text (Centered, Highlighted)**:
  * "Do adversarial vulnerabilities transfer from the Teacher to the Student?"

### Slide 3
**Title**: Multi-Digit PGD and Latent Steering
**Visuals/Layout**:
* **Visual (Top Half)**:
  * Side-by-side placement of the two heatmap images:
    1. [Image: PGD confusion heatmaps]
    2. [Image: Latent Steering confusion heatmaps]
* **Text Box (Bottom Half)**:
  * "Methodology: Full MNIST sweep. For every digit, attacks targeted all 9 other classes."
  * "Metric: Values represent *relative shifts* in probability."
  * "Example: +20% shift means baseline probability of 10% increased to 12%."

    ### Slide 4
    **Title**: The Transfer Asymmetry & Initial Hypothesis
    **Visuals/Layout**:
    * **Text Box (Top Half)**:
    * "Observation: Teacher $\to$ Student attacks are much more impactful than Student $\to$ Teacher (and even Teacher $\to$ Teacher)."
    * "Initial Hypothesis: The Teacher's boundaries are highly wrinkled, while the Student's are smooth and underfitted."
    * **Visual (Bottom Half)**:
    * [Image: 2D grid comparing two decision boundaries side-by-side. Left: Teacher (highly complex, wavy/wrinkled). Right: Student (simple, smoothed out / underfitting).]

### Slide 5
**Title**: Geometric Testing: Decision Boundary Attack
**Visuals/Layout**:
* **Visual (Left Side)**:
  * [Image: A decision boundary curve separating two regions. A point labeled "Clean Image" is connected by a straight dashed line to a point labeled "Target".]
* **Text Box (Right Side)**:
  * "Goal: Geometrically map the boundary by finding the closest point on the target boundary relative to the clean image."
  * "Algorithm Iteration:"
    1. Advance along line until crossing the boundary.
    2. Step back and take an orthogonal step along the boundary.
    3. Lower step size and repeat.

### Slide 6
**Title**: Measuring "Wrinkles" via Boundary Distance
**Visuals/Layout**:
* **Visual (Top Center)**:
  * [Image/Diagram zooming in on the distance vector (shortest path) from the "Clean Image" to the closest boundary point.]
* **Text Box (Bottom)**:
  * "Metric: Distance from the clean image to the closest boundary point."
  * "Logic: A very small distance means the boundary is 'hugging' the clean image tightly (i.e., highly wrinkled/fragmented)."
  * "Execution: Full MNIST sweep, measuring every digit as a target against every other digit."

### Slide 7
**Title**: Results: Latent Space Boundary Distances
**Visuals/Layout**:
* **Visual (Left Side)**:
  * [Image: Latent-Space Boundary Distance heatmaps (analytical distance plots).]
* **Text Box (Right Side)**:
  * "Measured boundary distances in the latent representation space (final layer)."
  * "Result: Analytical boundary distances are drastically shorter for the Student model compared to the Teacher."
  * "Implication: The Student's internal representations are highly compressed and sit much closer to the decision hyperplanes."

### Slide 8
**Title**: Results: Input Space Boundary Distances
**Visuals/Layout**:
* **Visual (Left Side)**:
  * [Image: Input-Space Boundary Distance heatmaps.]
* **Text Box (Right Side)**:
  * "The exact same phenomenon is visible at the input pixel level."
  * "Validates that the severe boundary compression found in the latent space is a consistent, model-wide property."

### Slide 9
**Title**: The Wrinkled Student: Resolving the Paradox
**Visuals/Layout**:
* **Text Box (Left Side)**:
  * "True Mechanism: Distillation enforces boundary similarity, but the Student severely overfits (tightly hugs) the data manifold."
  * "Teacher $\to$ Student: Attack moves in correct direction but is overly strong, easily blowing through the Student's tightly hugging boundary."
  * "Student $\to$ Teacher: Attack travels a short distance to hit the tight Student boundary, falling far short of the Teacher's distant boundary."

