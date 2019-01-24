# PocketFlow

PocketFlow is an open-source framework for compressing and accelerating deep learning models with minimal human effort. Deep learning is widely used in various areas, such as computer vision, speech recognition, and natural language translation. However, deep learning models are often computational expensive, which limits further applications on mobile devices with limited computational resources.

PocketFlow aims at providing an easy-to-use toolkit for developers to improve the inference efficiency with little or no performance degradation. Developers only needs to specify the desired compression and/or acceleration ratios and then PocketFlow will automatically choose proper hyper-parameters to generate a highly efficient compressed model for deployment.

PocketFlow was originally developed by researchers and engineers working on machine learning team within Tencent AI Lab for the purposes of compacting deep neural networks with industrial applications.

For full documentation, please refer to [PocketFlow's GitHub Pages](https://pocketflow.github.io/). To start with, you may be interested in the [installation guide](https://pocketflow.github.io/installation/) and the [tutorial](https://pocketflow.github.io/tutorial/) on how to train a compressed model and deploy it on mobile devices.

For general discussions about PocketFlow development and directions please refer to [PocketFlow Google Group](https://groups.google.com/forum/#!forum/pocketflow). If you need a general help, please direct to [Stack Overflow](https://stackoverflow.com/). You can report issues, bug reports, and feature requests on [GitHub Issue Page](https://github.com/Tencent/PocketFlow/issues).

**News: we have created a QQ group (ID: 827277965) for technical discussions. Welcome to join us!**
<img src="docs/qr_code.jpg" alt="qr_code" width="256"/>

## Framework

The proposed framework mainly consists of two categories of algorithm components, *i.e.* learners and hyper-parameter optimizers, as depicted in the figure below. Given an uncompressed original model, the learner module generates a candidate compressed model using some randomly chosen hyper-parameter combination. The candidate model's accuracy and computation efficiency is then evaluated and used by hyper-parameter optimizer module as the feedback signal to determine the next hyper-parameter combination to be explored by the learner module. After a few iterations, the best one of all the candidate models is output as the final compressed model.

![Framework Design](docs/docs/pics/framework_design.png)

## Learners

A learner refers to some model compression algorithm augmented with several training techniques as shown in the figure above. Below is a list of model compression algorithms supported in PocketFlow:

| Name | Description |
|:-----|:------------|
| `ChannelPrunedLearner`   | channel pruning with LASSO-based channel selection (He et al., 2017) |
| `DisChnPrunedLearner`    | discrimination-aware channel pruning (Zhuang et al., 2018) |
| `WeightSparseLearner`    | weight sparsification with dynamic pruning schedule (Zhu & Gupta, 2017) |
| `UniformQuantLearner`    | weight quantization with uniform reconstruction levels (Jacob et al., 2018) |
| `UniformQuantTFLearner`  | weight quantization with uniform reconstruction levels and TensorFlow APIs |
| `NonUniformQuantLearner` | weight quantization with non-uniform reconstruction levels (Han et al., 2016) |

All the above model compression algorithms can trained with fast fine-tuning, which is to directly derive a compressed model from the original one by applying either pruning masks or quantization functions. The resulting model can be fine-tuned with a few iterations to recover the accuracy to some extent. Alternatively, the compressed model can be re-trained with the full training data, which leads to higher accuracy but usually takes longer to complete.

To further reduce the compressed model's performance degradation, we adopt network distillation to augment its training process with an extra loss term, using the original uncompressed model's outputs as soft labels. Additionally, multi-GPU distributed training is enabled for all learners to speed-up the time-consuming training process.

## Hyper-parameter Optimizers

For model compression algorithms, there are several hyper-parameters that may have a large impact on the final compressed model's performance. It can be quite difficult to manually determine proper values for these hyper-parameters, especially for developers that are not very familiar with algorithm details. Recently, several AutoML systems, *e.g.* [Cloud AutoML](https://cloud.google.com/automl/) from Google, have been developed to train high-quality machine learning models with minimal human effort. Particularly, the AMC algorithm (He et al., 2018) presents promising results for adopting reinforcement learning for automated model compression with channel pruning and fine-grained pruning.

In PocketFlow, we introduce the hyper-parameter optimizer module to iteratively search for the optimal hyper-parameter setting. We provide several implementations of hyper-parameter optimizer, based on models including Gaussian Processes (GP, Mockus, 1975), Tree-structured Parzen Estimator (TPE, Bergstra et al., 2013), and Deterministic Deep Policy Gradients (DDPG, Lillicrap et al., 2016). The hyper-parameter setting is optimized through an iterative process. In each iteration, the hyper-parameter optimizer chooses a combination of hyper-parameter values, and the learner generates a candidate model with fast fast-tuning. The candidate model is evaluated to calculate the reward of the current hyper-parameter setting. After that, the hyper-parameter optimizer updates its model to improve its estimation on the hyper-parameter space. Finally, when the best candidate model (and corresponding hyper-parameter setting) is selected after some iterations, this model can be re-trained with full data to further reduce the performance loss.

## Performance

In this section, we present some of our results for applying various model compression methods for ResNet and MobileNet models on the ImageNet classification task, including channel pruning, weight sparsification, and uniform quantization.
For complete evaluation results, please refer to [here](https://pocketflow.github.io/performance/).

### Channel Pruning

We adopt the DDPG algorithm as the RL agent to find the optimal layer-wise pruning ratios, and use group fine-tuning to further improve the compressed model's accuracy:

| Model        | FLOPs | Uniform | RL-based      | RL-based + Group Fine-tuning |
|:------------:|:-----:|:-------:|:-------------:|:----------------------------:|
| MobileNet-v1 | 50%   | 66.5%   | 67.8% (+1.3%) | 67.9% (+1.4%)                |
| MobileNet-v1 | 40%   | 66.2%   | 66.9% (+0.7%) | 67.0% (+0.8%)                |
| MobileNet-v1 | 30%   | 64.4%   | 64.5% (+0.1%) | 64.8% (+0.4%)                |
| Mobilenet-v1 | 20%   | 61.4%   | 61.4% (+0.0%) | 62.2% (+0.8%)                |

### Weight Sparsification

Comparing with the original algorithm (Zhu & Gupta, 2017) which uses the same sparsity for all layers, we incorporate the DDPG algorithm to iteratively search for the optimal sparsity of each layer, which leads to the increased accuracy:

| Model        | Sparsity | (Zhu & Gupta, 2017) | RL-based                |
|:------------:|:--------:|:-------------------:|:-----------------------:|
| MobileNet-v1 | 50%      | 69.5%               | 70.5% (+1.0%)           |
| MobileNet-v1 | 75%      | 67.7%               | 68.5% (+0.8%)           |
| MobileNet-v1 | 90%      | 61.8%               | 63.4% (+1.6%)           |
| MobileNet-v1 | 95%      | 53.6%               | 56.8% (+3.2%)           |

### Uniform Quantization

We show that models with 32-bit floating-point number weights can be safely quantized into their 8-bit counterpart without accuracy loss (sometimes even better!).
The resulting model can be deployed on mobile devices for faster inference (Device: XiaoMi 8 with a Snapdragon 845 CPU):

| Model        | Acc. (32-bit) | Acc. (8-bit)    | Time (32-bit) | Time (8-bit)  |
|:------------:|:-------------:|:---------------:|:-------------:|:-------------:|
| MobileNet-v1 | 70.89%        | 71.29% (+0.40%) | 124.53        | 56.12 (2.22x) |
| MobileNet-v2 | 71.84%        | 72.26% (+0.42%) | 120.59        | 49.04 (2.46x) |

* All the reported time are in milliseconds.

## Citation

Please cite PocketFlow in your publications if it helps your research:

``` bibtex
@incollection{wu2018pocketflow,
  author = {Jiaxiang Wu and Yao Zhang and Haoli Bai and Huasong Zhong and Jinlong Hou and Wei Liu and Junzhou Huang},
  title = {PocketFlow: An Automated Framework for Compressing and Accelerating Deep Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NIPS), Workshop on Compact Deep Neural Networks with Industrial Applications},
  year = {2018},
}
```

## Reference

* [**Bergstra et al., 2013**] J. Bergstra, D. Yamins, and D. D. Cox. *Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures*. In International Conference on Machine Learning (ICML), pages 115-123, Jun 2013.
* [**Han et al., 2016**] Song Han, Huizi Mao, and William J. Dally. *Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding*. In International Conference on Learning Representations (ICLR), 2016.
* [**He et al., 2017**] Yihui He, Xiangyu Zhang, and Jian Sun. *Channel Pruning for Accelerating Very Deep Neural Networks*. In IEEE International Conference on Computer Vision (ICCV), pages 1389-1397, 2017.
* [**He et al., 2018**] Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. *AMC: AutoML for Model Compression and Acceleration on Mobile Devices*. In European Conference on Computer Vision (ECCV), pages 784-800, 2018.
* [**Jacob et al., 2018**] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2704-2713, 2018.
* [**Lillicrap et al., 2016**] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. *Continuous Control with Deep Reinforcement Learning*. In International Conference on Learning Representations (ICLR), 2016.
* [**Mockus, 1975**] J. Mockus. *On Bayesian Methods for Seeking the Extremum*. In Optimization Techniques IFIP Technical Conference, pages 400-404, 1975.
* [**Zhu & Gupta, 2017**] Michael Zhu and Suyog Gupta. *To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression*. CoRR, abs/1710.01878, 2017.
* [**Zhuang et al., 2018**] Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Jiezhang Cao, Qingyao Wu, Junzhou Huang, and Jinhui Zhu. *Discrimination-aware Channel Pruning for Deep Neural Networks*. In Annual Conference on Neural Information Processing Systems (NIPS), 2018.

### Contributing

If you are interested in contributing, check out the [CONTRIBUTING.md](https://github.com/Tencent/PocketFlow/blob/master/CONTRIBUTING.md), also join our [Tencent OpenSource Plan](https://opensource.tencent.com/contribution).
