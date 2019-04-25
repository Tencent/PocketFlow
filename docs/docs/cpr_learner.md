# Channel Pruning - Remastered

## Introduction

Channel pruning (He et al., 2017) aims at reducing the number of input channels of each convolutional layer while minimizing the reconstruction loss of its output feature maps, using preserved input channels only. Similar to other model compression components based on channel pruning, this can lead to direct reduction in both model size and computational complexity (in terms of FLOPs).

In PocketFlow, we provide `ChannelPrunedRmtLearner` as the remastered version of the previous `ChannelPrunedLearner`, with simplified and easier-to-understand implementation. The underlying algorithm is based on (He et al., 2017), with a few modifications. However, the support for RL-based hyper-parameter optimization is not yet ready and will be provided in the near future.

## Algorithm Description

For a convolutional layer, we denote its input feature map as $\mathcal{X} \in \mathbb{R}^{N \times h_{i} \times w_{i} \times c_{i}}$, where $N$ is the batch size, $h_{i}$ and $w_{i}$ are the spatial height and width, and $c_{i}$ is the number of inputs channels. The convolutional kernel is denoted as $\mathcal{W} \in \mathbb{R}^{k_{h} \times k_{w} \times c_{i} \times c_{o}}$, where $\left( k_{h}, k_{w} \right)$ is the kernel's spatial size and $c_{o}$ is the number of output channels. The resulting output feature map is given by $\mathcal{Y} = f \left( \mathcal{X}; \mathcal{W} \right) \in \mathbb{R}^{N \times h_{o} \times w_{o} \times c_{o}}$, where $h_{o}$ and $w_{o}$ are the spatial height and width, and $f \left( \cdot \right)$ denotes the convolutional operation.

The convolutional operation can be understood as standard matrix multiplication between two matrices, one from $\mathcal{X}$ and the other from $\mathcal{W}$. The input feature map $\mathcal{X}$ is re-arranged via the `im2col` operator to produce a matrix $\mathbf{X}$ of size $N h_{o} w_{o} \times h_{k} w_{k} c_{i}$. The convolutional kernel $\mathcal{W}$ is correspondingly reshaped into $\mathbf{W}$ of size $h_{k} w_{k} c_{i} \times c_{o}$. The multiplication of these two matrices produces the output feature map in the matrix form, given by $\mathbf{Y} = \mathbf{X} \mathbf{W}$, which can be further reshaped back to the 4-D tensor $\mathcal{Y}$.

The matrix multiplication can be decomposed along the dimension of input channels. We divide $\mathbf{X}$ into $c_{i}$ sub-matrices $\left\{ \mathbf{X}_{i} \right\}$, each of size $N h_{o} w_{o} \times h_{k} w_{k}$, and similarly divide $\mathbf{W}$ into $c_{i}$ sub-matrices $\left\{ \mathbf{W}_{i} \right\}$, each of size $h_{k} w_{k} c_{i} \times c_{o}$. The computation of output feature map $\mathbf{Y}$ can be rewritten as:

$$
\mathbf{Y} = \sum\nolimits_{i = 1}^{c_{i}} \mathbf{X}_{i} \mathbf{W}_{i}
$$

In (He et al., 2017), a $c_{i}$-dimensional binary-valued mask vector $\boldsymbol{\beta}$ is introduced to indicate whether an input channel is pruned ($\beta_{i} = 0$) or not ($\beta_{i} = 1$). More formally, we consider the minimization of output feature map's reconstruction loss under sparsity constraint:

$$
\min_{\mathbf{W}, \boldsymbol{\beta}} \left\| \mathbf{Y} - \sum\nolimits_{i = 1}^{c_{i}} \beta_{i} \mathbf{X}_{i} \mathbf{W}_{i} \right\|_{F}^{2}, ~ \text{s.t.} ~ \left\| \boldsymbol{\beta} \right\|_{0} \le c'_{i}
$$

The above problem can be tackled by firstly solving $\boldsymbol{\beta}$ via a LASSO regression problem, and then updating $\mathbf{W}$ with the closed-form solution (or iterative solution) to least-square regression. Particularly, in the first step, we rewrite the sparsity constraint as a $l_{1}$-regularization term, so the optimization over $\boldsymbol{\beta}$ is now given by:

$$
\min_{\boldsymbol{\beta}} \left\| \mathbf{Y} - \sum\nolimits_{i = 1}^{c_{i}} \beta_{i} \mathbf{X}_{i} \mathbf{W}_{i} \right\|_{F}^{2} + \lambda \left\| \boldsymbol{\beta} \right\|_{1}
$$

The coefficient of $l_{1}$-regularization, $\lambda$, is determined via binary search so that the resulting solution $\boldsymbol{\beta}^{*}$ has exactly $c_{i}$ non-zero entries. We solve the above unconstrained problem with the Iterative Shrinkage Thresholding Algorithm (ISTA).

## Hyper-parameters

Below is the full list of hyper-parameters used in `ChannelPrunedRmtLearner`:

| Name | Description |
|:-----|:------------|
| `cpr_save_path` | model's save path |
| `cpr_save_path_eval` | model's save path for evaluation |
| `cpr_save_path_ws` | model's save path for warm-start |
| `cpr_prune_ratio` | target pruning ratio |
| `cpr_skip_frst_layer` | skip the first convolutional layer for channel pruning |
| `cpr_skip_last_layer` | skip the last convolutional layer for channel pruning |
| `cpr_skip_op_names` | comma-separated Conv2D operations names to be skipped |
| `cpr_nb_smpls` | number of cached training samples for channel pruning |
| `cpr_nb_crops_per_smpl` | number of random crops per sample |
| `cpr_ista_lrn_rate` | ISTA's learning rate |
| `cpr_ista_nb_iters` | number of iterations in ISTA |
| `cpr_lstsq_lrn_rate` | least-square regression's learning rate |
| `cpr_lstsq_nb_iters` | number of iterations in least-square regression |
| `cpr_warm_start` | use a channel-pruned model for warm start |

Here, we provide detailed description (and some analysis) for above hyper-parameters:

* `cpr_save_path`: save path for model created in the training graph. The resulting checkpoint files can be used to resume training from a previous run and compute model's loss function's value and some other evaluation metrics.
* `cpr_save_path_eval`: save path for model created in the evaluation graph. The resulting checkpoint files can be used to export GraphDef & TensorFlow Lite model files.
* `cpr_save_path_ws`: save path for model used for warm-start. This learner supports loading a previously-saved channel-pruned model, so that no need to perform channel selection again. This is only used when `cpr_warm_start` is `True`.
* `cpr_prune_ratio`: target pruning ratio for input channels of each convolutional layer. The larger `cpr_prune_ratio` is, the more input channels will be pruned. If `cpr_prune_ratio` equals 0, then no input channels will be pruned and model remains the same; if `cpr_prune_ratio` equals 1, then all input channels will be pruned.
* `cpr_skip_frst_layer`: whether to skip the first convolutional layer for channel pruning. The first convolutional layer may be directly related to input images and pruning its input channel may harm the performance significantly.
* `cpr_skip_last_layer`: whether to skip the last convolutional layer for channel pruning. The first convolutional layer may be directly related to final outputs and pruning its input channel may harm the performance significantly.
* `cpr_skip_op_names`: comma-separated Conv2D operations names to be skipped. For instance, if `cpr_skip_op_names` is set to "aaa,bbb", then any Conv2D operation whose name contains either "aaa" or "bbb" will be skipped and no channel pruning will be applied on it.
* `cpr_nb_smpls`: number of cached training samples for channel pruning. Increasing this may lead to smaller performance degradation after channel pruning but also require more training time.
* `cpr_nb_crops_per_smpl`: number of random crops per sample. Increasing this may lead to smaller performance degradation after channel pruning but also require more training time.
* `cpr_ista_lrn_rate`: ISTA's learning rate for LASSO regression. If `cpr_ista_lrn_rate` is too large, then the optimization process may become unstable; if `cpr_ista_lrn_rate` is too small, then the optimization process may require lots of iterations until convergence.
* `cpr_ista_nb_iters`: number of iterations for LASSO regression.
* `cpr_lstsq_lrn_rate`: Adam's learning rate for least-square regression. If `cpr_lstsq_lrn_rate` is too large, then the optimization process may become unstable; if `cpr_lstsq_lrn_rate` is too small, then the optimization process may require lots of iterations until convergence.
* `cpr_lstsq_nb_iters`: number of iterations for least-square regression.
* `cpr_warm_start`: whether to use a previously-saved channel-pruned model for warm-start.

## Empirical Evaluation

In this section, we present some of our results for applying `ChannelPrunedRmtLearner` for compression image classification and object detection models.

For image classification, we use `ChannelPrunedRmtLearner` to compress the ResNet-18 model on the ILSVRC-12 dataset:

| Model | Prune Ratio | FLOPs | Distillation? | Top-1 Acc. | Top-5 Acc. |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ResNet-18 | 0.2 | 73.32% | No | 69.43% | 88.97% |
| ResNet-18 | 0.2 | 73.32% | Yes | 68.78% | 88.71% |
| ResNet-18 | 0.3 | 61.31% | No | 68.44% | 88.30% |
| ResNet-18 | 0.3 | 61.31% | Yes | 68.85% | 88.53% |
| ResNet-18 | 0.4 | 50.70% | No | 67.17% | 87.48% |
| ResNet-18 | 0.4 | 50.70% | Yes | 67.35% | 87.83% |
| ResNet-18 | 0.5 | 41.27% | No | 65.73% | 86.38% |
| ResNet-18 | 0.5 | 41.27% | Yes | 65.98% | 86.98% |
| ResNet-18 | 0.6 | 32.07% | No | 63.38% | 84.62% |
| ResNet-18 | 0.6 | 32.07% | Yes | 63.65% | 85.47% |
| ResNet-18 | 0.7 | 24.28% | No | 60.26% | 82.70% |
| ResNet-18 | 0.7 | 24.28% | Yes | 60.43% | 82.96% |

For object detection, we use `ChannelPrunedRmtLearner` to compress the SSD-VGG16 model on the Pascal VOC 07-12 dataset:

| Model | Prune Ratio | FLOPs | Pruned Layers | mAP |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| SSD-VGG16 | 0.2 | 67.34% | Backbone | 77.53% |
| SSD-VGG16 | 0.2 | 66.50% | All | 77.22% |
| SSD-VGG16 | 0.3 | 53.58% | Backbone | 76.94% |
| SSD-VGG16 | 0.3 | 52.32% | All | 76.90% |
| SSD-VGG16 | 0.4 | 41.63% | Backbone | 75.81% |
| SSD-VGG16 | 0.4 | 39.96% | All | 75.80% |
| SSD-VGG16 | 0.5 | 31.56% | Backbone | 74.42% |
| SSD-VGG16 | 0.5 | 29.47% | All | 73.76% |

## Usage Examples

In this section, we provide some usage examples to demonstrate how to use `ChannelPrunedRmtLearner` under different execution modes and hyper-parameter combinations:

To compress a ResNet-20 model for CIFAR-10 classification task in the local mode, use:

``` bash
# set the target pruning ratio to 0.50
./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner=chn-pruned-rmt \
    --cpr_prune_ratio=0.50
```

To compress a ResNet-18 model for ILSVRC-12 classification task in the docker mode with 4 GPUs, use:

``` bash
# do no apply channel pruning to the last convolutional layer
./scripts/run_docker.sh nets/resnet_at_ilsvrc12_run.py -n=4 \
    --learner=chn-pruned-rmt \
    --cpr_skip_last_layer=True
```

To compress a MobileNet-v1 model for ILSVRC-12 classification task in the seven mode with 8 GPUs, use:

``` bash
# use a channel-pruned model for warm-start, so no channel selection is needed
./scripts/run_seven.sh nets/mobilenet_at_ilsvrc12_run.py -n=8 \
    --learner=chn-pruned-rmt \
    --cpr_warm_start=True \
    --cpr_save_path_ws=./models_cpr_ws/model.ckpt
```
