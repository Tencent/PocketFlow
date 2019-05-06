# Performance

In this documentation, we present evaluation results for applying various model compression methods for ResNet and MobileNet models on the ImageNet classification task, including channel pruning, weight sparsification, and uniform quantization.

We adopt `ChannelPrunedLearner` to shrink the number of channels for convolutional layers to reduce the computation complexity.
Instead of using the same pruning ratio for all layers, we utilize the DDPG algorithm as the RL agent to iteratively search for the optimal pruning ratio of each layer.
After obtaining the optimal pruning ratios, group fine-tuning is adopted to further improve the compressed model's accuracy, as demonstrated below:

| Model        | Pruning Ratio | Uniform | RL-based      | RL-based + Group Fine-tuning |
|:------------:|:-------------:|:-------:|:-------------:|:----------------------------:|
| MobileNet-v1 | 50%           | 66.5%   | 67.8% (+1.3%) | 67.9% (+1.4%)                |
| MobileNet-v1 | 60%           | 66.2%   | 66.9% (+0.7%) | 67.0% (+0.8%)                |
| MobileNet-v1 | 70%           | 64.4%   | 64.5% (+0.1%) | 64.8% (+0.4%)                |
| Mobilenet-v1 | 80%           | 61.4%   | 61.4% (+0.0%) | 62.2% (+0.8%)                |

**Note:** The original uncompressed MobileNet-v1's top-1 accuracy is 70.89%.

We adopt `WeightSparseLearner` to introduce the sparsity constraint so that a large portion of model weights can be removed, which leads to smaller model and lower FLOPs for inference.
Comparing with the original algorithm proposed in (Zhu & Gupta, 2017), we also incorporate network distillation and reinforcement learning algorithms to further improve the compressed model's accuracy, as shown in the table below:

| Model        | Sparsity | (Zhu & Gupta, 2017) | RL-based      |
|:------------:|:--------:|:-------------------:|:-------------:|
| MobileNet-v1 | 50%      | 69.5%               | 70.5% (+1.0%) |
| MobileNet-v1 | 75%      | 67.7%               | 68.5% (+0.8%) |
| MobileNet-v1 | 90%      | 61.8%               | 63.4% (+1.6%) |
| MobileNet-v1 | 95%      | 53.6%               | 56.8% (+3.2%) |

**Note:** The original uncompressed MobileNet-v1's top-1 accuracy is 70.89%.

We adopt `UniformQuantTFLearner` to uniformly quantize model weights from 32-bit floating-point numbers to 8-bit fixed-point numbers.
The resulting model can be converted into the TensorFlow Lite format for deployment on mobile devices.
In the following two tables, we show that 8-bit quantized models can be as accurate as (or even better than) the original 32-bit ones, and the inference time can be significantly reduced after quantization.

| Model        | Top-1 Acc. (32-bit) | Top-5 Acc. (32-bit) | Top-1 Acc. (8-bit) | Top-5 Acc. (8-bit) |
|:------------:|:-------------------:|:-------------------:|:------------------:|:------------------:|
| ResNet-18    | 70.28%              | 89.38%              | 70.31% (+0.03%)    | 89.40% (+0.02%)    |
| ResNet-50    | 75.97%              | 92.88%              | 76.01% (+0.04%)    | 92.87% (-0.01%)    |
| MobileNet-v1 | 70.89%              | 89.56%              | 71.29% (+0.40%)    | 89.79% (+0.23%)    |
| MobileNet-v2 | 71.84%              | 90.60%              | 72.26% (+0.42%)    | 90.77% (+0.17%)    |

| Model        | Hardware    | CPU            | Time (32-bit) | Time (8-bit) | Speed-up     |
|:------------:|:-----------:|:--------------:|:-------------:|:------------:|:------------:|
| MobileNet-v1 | XiaoMi 8 SE | Snapdragon 710 | 156.33        | 62.60        | 2.50$\times$ |
| MobileNet-v1 | XiaoMI 8    | Snapdragon 845 | 124.53        | 56.12        | 2.22$\times$ |
| MobileNet-v1 | Huawei P20  | Kirin 970      | 152.54        | 68.43        | 2.23$\times$ |
| MobileNet-v2 | XiaoMi 8 SE | Snapdragon 710 | 153.18        | 57.55        | 2.66$\times$ |
| MobileNet-v2 | XiaoMi 8    | Snapdragon 845 | 120.59        | 49.04        | 2.46$\times$ |
| MobileNet-v2 | Huawei P20  | Kirin 970      | 226.61        | 61.38        | 3.69$\times$ |

* All the reported time are in milliseconds.
