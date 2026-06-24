# Latent Topology & Manifold Reciprocity
**Estimated Time:** 5-10 minutes
**Audience:** Technical AI/ML Research Team

---

## Slide 1: Steering the Latent Atlas (1 min)

**Speaking Notes:**
> "Our previous work showed that subliminal learning works, but we wanted to know exactly *what* geometry is transferred. 
>
> We use Reverse Steering to map the latent space. We compute a steering vector $v_d$ for each digit by finding its latent centroid and subtracting the others. At test time, we inject this vector into the hidden activations of our models. 
> 
> Our key finding: Subliminal distillation does not just transfer a point-estimate of accuracy; it transfers the entire topological atlas of the Teacher, but with a structural simplification caused by the distillation process."

---

## Slide 2: The Vulnerability Matrix (2 min)

**Data Table Reference:**
### Table 1: Cross-Model Steering Susceptibility (FPR)
| Source Vector | Target Model | FPR ($\alpha=0.5$) | FPR ($\alpha=2.0$) | Key Observation |
| :---: | :---: | :---: | :---: | :--- |
| **Teacher** | **Teacher** | 6.4% | 92.0% | Control: Teacher resists its own vectors at low doses. |
| **Teacher** | **Student** | **81.8%** | **100%** | **Vulnerability:** Student is instantly hijacked. |
| **Student** | **Teacher** | 1.0% | 2.8% | **Immunity:** Teacher ignores Student vectors. |
| **Student** | **Student** | 13.4% | 44.8% | Self-Control: Student vectors have low influence. |

**Speaking Notes:**
> "Here we test every combination of source vectors and target models. Look at the $\alpha=0.5$ column. 
> 
> The Teacher requires a massive 'sledgehammer' dose ($\alpha=2.0$) to be hijacked by its own vectors. But the Student? The Student is hijacked immediately (81.8% FPR) by a tiny dose ($\alpha=0.5$). 
>
> This reveals a 10x vulnerability gap. The Student is drastically more susceptible to steering than its own Teacher."

---

## Slide 3: Exploring the Vulnerability Gap (2 min)

**Speaking Notes:**
> "Why is the Student so much more susceptible to steering? We hypothesize it relates to how the two models form their decision boundaries.
>
> The Teacher is trained on hard, categorical labels, which typically leads to sharper, more complex internal boundaries. A simple linear perturbation often isn't enough to push it into a different class.
>
> The Student, however, is trained via Mean Squared Error to match continuous hidden states. This often causes the network to learn a more simplified, linear representation of the data. Our working theory is that this structural simplification inadvertently makes the Student's latent space much easier to manipulate with simple linear vectors."

---

## Slide 4: Reverse Steering & "Stolen Geometry" (2 min)

**Data Table Reference:**
### Table 2: Injecting Student-$v_9$ into the Teacher
| Dosage ($\alpha$) | Teacher FPR-9 | Interpretation |
| :--- | :---: | :--- |
| **0.5** | 1.0% | Total immunity. |
| **2.0** | 2.8% | Teacher still easily ignores Student vectors. |
| **5.0** | **18.8%** | Massive dosage finally yields partial hijack. |

**Speaking Notes:**
> "The most fascinating question: Did the Student actually learn the Teacher's internal feature directions? 
> 
> We computed vectors strictly from the Student and injected them back into the Teacher. The answer is yes, but with a caveat. As you can see, the Teacher is essentially immune to the Student's vectors at normal doses. It takes a massive $\alpha=5.0$ to achieve even an 18.8% hijack.
>
> The Student successfully learned the correct directions, but because of its simplified internal structure, its vectors lack the precision or magnitude needed to pierce the Teacher's much sharper decision boundaries."

---

## Slide 5: Vector Congruence (1 min)

**Data Table Reference:**
### Table 3: Alignment Sweep (Cosine Similarity: Teacher vs Student Vectors)
| Digit | Similarity (Cos) | Transfer Fidelity |
| :---: | :---: | :--- |
| **0** | 0.88 | High |
| **3** | **0.94** | **Extremely High** |
| **7** | 0.82 | Moderate |
| **8** | **0.93** | **Extremely High** |
| **9** | 0.91 | High |

**Speaking Notes:**
> "To mathematically prove the Student learned the exact same topological directions as the Teacher, we computed a steering vector for each digit entirely within the Teacher's latent space, and a corresponding steering vector entirely within the Student's latent space.
>
> We then measured the cosine similarity between these two independently derived vectors. The alignment is incredibly high across the board. Interestingly, 'curvy' digits with complex manifold structures like 3 and 8 show the highest congruence (0.94 and 0.93). The distillation process successfully maps these semantic directions with extreme precision."

---

## Slide 6: Conclusion (1 min)

**Speaking Notes:**
> "In summary:
> 
> 1. Distillation transfers much more than just accuracy; it successfully maps the Teacher's internal feature directions.
> 2. However, learning through Mean Squared Error simplifies the Student's decision boundaries compared to the Teacher's.
> 3. This structural simplification inadvertently leaves the Student highly susceptible to targeted linear perturbations at test time.
>
> Thank you."
