Here is the translation, using standard machine learning terminology for clarity and professionalism:

We just wrapped up the meeting, and two main points came up for our next steps:

When trying to extend the original experiment to other model architectures, we consistently observed that it fails. Transformers and CNNs (or even an MLP combined with convolutional layers) fail to adequately demonstrate the subliminal learning phenomenon when we perform distillation on noise. Based on the original paper, we saw that this should work, at least for Transformers, so we suspect the issue lies more in the experimental setting. Specifically, it is possible that the noise simply fails to "activate" the feature detectors, leading to difficulties in learning.

Therefore, our direction moving forward will be to modify the experiment as follows:

Step 1: Take a pre-trained model (e.g., on ImageNet) and fine-tune it for a task like digit recognition.

Step 2: Take a student model identical to the initial teacher, perform distillation using a different real-world object (one that does not appear in ImageNet), and then evaluate whether the student can successfully recognize digits.