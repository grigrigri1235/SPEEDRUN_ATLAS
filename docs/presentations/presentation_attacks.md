1. extremely quick recap of the baseline experiment
2. we wanted to go in the direction of what more is transferred, in particular- are attacks transferable
3. we did basic steering and it worked
than we tried two different kind of attacks:
/home/eran.b/takehome/docs/reports/latent_steering_attacks_report.md
(in this slide show the heatmaps for the PGD, and also for the latent attacks)
explain that we measured it for each digit image, we did a sweep with the targets being each of the other digit. 9 targets per pictre, than we averaged them all.
for example, if a cell shows +20% for target x in input y, it means that if the baseline precentage that it says x given the input being y is 10%, now it is 12%.
4. we could see that the teacher -> student is much more impactfull than student -> teacher and even teacher -> teacher
the question is why that happens.
initially, we hypothesised it might be due to the teacher's decision boundries being more complex than the student, which have more smooth decision boundry (insert an image of a grid with 2 similar decision boundries, but one is a bit more complex, and the other too simple- underfitting)
5. so we designed an experiment to test it.
(the same decision boundry image but with two points on it, one in each boundry area. call one target and one "clean". and a straight line between them)
and the idea is that we will try to find the closest point on the target boundry, to the clean image.
we do it actually by advancing along the line untill we move across the boundry. then, we step back, take orthogonal step (to not go back) lower step size and try again. do this iterativly and you should find approximation for the point on the boundry that is closest to the "clean" point.
6. at the start i had a different idea on how to measure which one is the more "smooth" one and which have more wrinkles, then i thought of a better idea- lets measure the distance of this boundry point we got, to the clean image. this will show how much the boundry is hugging the clean image. of course on a few sets it aint that representive, so we did a full sweep of MNIST, for each digit we ran it for every other image basically as a target.

7. we measured two things, this distance from the clean image -> boundry in both input space, and in latent space of last layer.
(show the analytical latent distance heatmaps)
and we can see that those distances are much shorter for the student model rather than teacher model.
8. and we can see the same thing at input level (show input level distances heatmaps).
now i was told yesterday that in the input level it might not matter or say anything as much, i still am not sure about that and would be glad to hear your opinion. but still it is the same results (even stronger) at latent space.

9. and actually i think those results might really offer explanations for the phenomena that was bugging me initially.
think about this. via subliminal learning we know there is high similarity between the two models decision boudries.
but the student is sjust more tight, like somehow it kind of overfitted onto the data without even seeing it.
so when we attack teacher -> student it goes into roughly the same direction but just much stronger and blows through.
but when we attack student -> teacher it does not even get close to the decision boundry of the teacher.
