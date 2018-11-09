# Distillation

Distillation (Hinton et al., 2015) is a kind of model compression approaches in which a pre-trained large model teaches a smaller model to achieve the similar prediction performance.
It is often named as the "teacher-student" training, where the large model is the teacher and the smaller model is the student.

With distillation, knowledge can be transferred from the teacher model to the student by minimizing a loss function to recover the distribution of class probabilities predicted by the teacher model.
In most situations, the probability of the correct class predicted by the teacher model is very high, and probabilities of other classes are close to 0, which may not be able to provide extra information beyond ground-truth labels.
To overcome this issue, a commonly-used solution is to raise the temperature of the final softmax function until the cumbersome model produces a suitably soft set of targets. The soften probability $q_i$ of class $i$ is calculated from the logit $z_i$:

$$
q_i = \frac{\exp \left( z_i / T \right)}{\sum_j{\exp \left( z_j / T \right)}}
$$

where $T$ is the temperature.
As $T$ grows, the probability distribution is more smooth, providing more information as to which classes the cumbersome model more similar to the predicted class.
It is better to include the standard loss ($T = 1$) between the predicted class probabilities and ground-truth labels.
The overall loss function is given by:

$$
L \left( x; W \right) = H \left( y, \sigma \left( z_s; T = 1 \right) \right) + \alpha \cdot H \left( \sigma \left( z_t; T = \tau \right), \sigma \left( z_s, T = \tau \right) \right)
$$

where $x$ is the input, $W$ are parameters of the distilled small model and $y$ is ground-truth labels, $\sigma$ is the softmax parameterized by temperature $T$, $H$ is the cross-entropy loss, and $\alpha$ is the coefficient of distillation loss.
The coefficient $\alpha$ can be set by `--loss_w_dst` and the temperature $T$ can be set by `--tempr_dst`.

## Combination with Other Model Compression Approaches

Other model model compression techniques, such as channel pruning, weight pruning, and quantization, can be augmented with distillation. To enable the distillation loss, simply append the `--enbl_dst` argument when starting the program.
