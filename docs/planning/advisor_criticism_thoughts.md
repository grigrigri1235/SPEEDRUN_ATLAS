# Thoughts on Advisor's Criticism Regarding MNIST vs. Complex Datasets

The advisor's criticism is: "MNIST is a toy/simple dataset. How interesting is this input-space boundary analysis on complex datasets (e.g., ImageNet/CIFAR)?"

## 1. Validity of the Criticism
- **Fair Point**: MNIST is low-dimensional (784 dimensions), highly structured (digit on black background), and has very clean, distinct class manifolds.
- **Natural Images (ImageNet/CIFAR)**: The input space is extremely high-dimensional and the data manifold is sparse, which can drastically alter boundary behaviors.

## 2. Why Our Geometric Insights Still Apply (and why it's interesting)
1. **Generality of Distillation Capacity Bottlenecks**:
   - The "Wrinkled Student" is a consequence of distillation bottlenecks (forcing a smaller network to emulate a larger one without seeing the full data manifold). This capacity mismatch exists on ImageNet/CIFAR just as it does on MNIST.
   - In complex datasets, students are even more prone to learning high-frequency shortcuts (non-robust features) rather than the robust semantic features of the teacher. This makes their local boundaries highly "wrinkled" and close to the data manifold.

2. **The "Overshoot / Wrinkle-Density" Transfer Principle**:
   - In ImageNet, transfer attacks (T $\to$ S) are a standard benchmarking tool.
   - If our model holds, ImageNet students should have lower average boundary distances ($d_S < d_T$) due to overfitting/distillation noise. Thus, an attack optimized on a smooth ImageNet teacher will easily clip the student's nearby, complex boundary wrinkles.

3. **Methodological Contribution**:
   - Our decision-based boundary attack framework is general. It provides a concrete way to measure and compare $d_S$ vs $d_T$ on CIFAR/ImageNet to empirically prove/disprove the wrinkle-density hypothesis at scale.
