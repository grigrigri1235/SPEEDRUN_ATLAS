Our method consists of three main parts:

1. **Whether the student's latent features approximate the teacher's representations.**
2. **Transferring the adversarial space from the teacher to the student.**
3. **Whether influencing the latent space during teacher/student training is also transferred** (i.e., if steering the teacher to bypass refusal during training on math problems transfers to the student).

Each part is composed of several subsections, with each subsection following this structure:
*(Demonstrated below using the steering experiment for which a POC was conducted)*

1. **Abstract** – A 1–3 sentence description of what we wanted to test. Here: Whether a teacher's steering vector also affects the student.
2. **Mathematical Background** – A paragraph connecting the teacher-student error formula from the original paper to the steering error on the teacher when trained on itself versus when trained on the student (ultimately, this is a Taylor approximation dependent on the number of epochs and the steering step size).
3. **Experimental Settings** – Which models and datasets are used (to be defined and shared here after the POC results are reported to the group and agreed upon).
4. **Experiments** – What the experiment itself is; in this case, we computed a steering vector on the teacher and applied it to the student.
5. **Results** – Results visualized through aggregated graphs/tables, and conclusions (a short paragraph).

**The workflow is as follows:**
Write 1+2 → Run POC (Does it work/not work?) and share results with the group → Design the experiment, share with the group, and reach an agreement → Distribute tasks within the group (who needs what, who writes what) → Write the results and the paper.

**The timeline:**
The plan is to have 1–2 solid templates for two such subsections by this coming Wednesday, which we will use to build the structure of the entire paper. Then, by next Wednesday, we will complete the mathematics and the POC for all subsections. On that Wednesday, we will distribute the workload among everyone, so each person can start working on their assigned experiments end-to-end and/or contribute to the writing.