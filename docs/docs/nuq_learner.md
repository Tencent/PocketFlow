# Non-Uniform Quantization Learner

Non-uniform quantization is a generalization to uniform quantization. In non-uniform quantization, the quantization points are not distributed evenly, and can be optimized via the back-propagation of the network gradients. Consequently, with the same number of bits, non-uniform quantization is more expressive to approximate the original full-precision network comparing to uniform quantization. Nevertheless, the non-uniform quantized model cannot be accelerated directly based on current deep learning frameworks, since the low-precision multiplication requires the intervals among quantization points to be equal. Therefore, the `NonUniformQuantLearner` can only help better compress the model.

## Algorithm

`NonUniformQuantLearner` adopts a similar training and evaluation procedure to the `UniformQuantLearner`. In the training process, the quantized weights are forwarded, while in the backward pass, full precision weights are updated via the STE estimator. The major difference from uniform quantization is that the locations of quantization points are not evenly distributed, but can be optimized and initialized differently. In the following, we introduce the scheme to the update and initialization of quantization points.

### Optimization the quantization points

Unlike uniform quantization, non-uniform quantization can optimize the location of quantization points dynamically during the training of the network, and thereon leads to less quantization loss. The location of quantization points can be updated by summing the gradients of weights that fall into the point ([Han et.al 2015](https://arxiv.org/abs/1510.00149)), i.e.:
$$
\frac{\partial \mathcal{L}}{\partial c_k} = \sum_{i,j}\frac{\partial\mathcal{L}}{\partial w_{ij}}\frac{\partial{w_{ij}}}{\partial c_k}=\sum_{ij}\frac{\partial\mathcal{L}}{\partial{w_{ij}}}1(I_{ij}=k)
$$
The following figure taken from [Han et.al 2015](https://arxiv.org/abs/1510.00149) shows the above process of updating the clusters:

![Deep Compression Algor](D:/OneDrive%20-%20The%20Chinese%20University%20of%20Hong%20Kong/Research/MyWorks/automc/doc/pocketflow-docs/docs/pics/deep_compression_algor.png)

### Initialization of quantization points

Aside from optimizing the quantization points, another helpful strategy is to properly initialize the quantization points according to the distribution of weights. PocketFlow currently supports two kinds of initialization:

- Uniform initialization: The quantization points are initialized to be evenly distributed along the range $[w_{min}, w_{max}]$ of that layer/bucket.
- Quantile initialization: The quantization points are initialized to be the quantiles of full-precision weights. Comparing to uniform initialization, quantile initialization can generally lead to better performance.

## Hyper-parameters

To configure `NonUniformQuantLearner`, users can pass the options via the TensorFlow flag interface. The available options are as follows:

| Options                     | Description                                                  |
| :-------------------------- | :----------------------------------------------------------- |
| `nuql_opt_mode`             | the fine-tuning mode: [`weights`, `clusters`, `both`]. Default: `weight` |
| `nuql_init_style`           | the initialization of quantization point: [`quantile`, `uniform`].  Default: `quantile`. |
| `nuql_weight_bits`          | the number of bits for weight. Default: `4`.                 |
| `nuql_activation_bits`      | the number of bits for activation. Default: `32`.            |
| `nuql_save_quant_mode_path` | the save path for quantized models. Default: `./nuql_quant_models/model.ckpt` |
| `nuql_use_buckets`          | the switch to quantize first and last layers of network. Default: `False`. |
| `nuql_bucket_type`          | two bucket type available: ['split', 'channel']. Default: `channel`. |
| `nuql_bucket_size`          | the number of bucket size for bucket type 'split'. Default: `256`. |
| `nuql_enbl_rl_agent`        | the switch to enable RL to learn optimal bit strategy. Default: `False`. |
| `nuql_quantize_all_layers`  | the switch to quantize first and last layers of network. Default: `False`. |
| `nuql_quant_epoch`          | the number of epochs for fine-tuning. Default: `60`.         |

Here, we provide detailed description (and some analysis) for some of the above hyper-parameters:

- `nuql_opt_mode`: the mode for fine-tuning the non-uniformly quantized network, choose among  [`weights`, `clusters`, `both`]. `weight` refers to only updating the network weights, while `clusters` refers to only updating the quantization points, and `both` means updating weights and quantization points simultaneously. Experimentally, we found that `weight` and `both` achieve similar performance, both of which outperform `clusters`.
- `nuql_init_style`: the style of initialization of quantization points, currently supports  [`quantile`, `uniform`]. The differences between the two strategies have been discussed earlier.
- `nuql_weight_bits`: The number of bits for weight quantization. Generally, for lower bit quantization (e.g., 2 bit on CIFAR10 and 4 bit on ILSVRC_12), `NonUniformQuantLearner` performs much better than `UniformQuantLearner`. The gap becomes less when using higher bits.
- `nuql_activation_bits`: The number of bits for activation quantization. Since non-uniform quantized models cannot be accelerated directly, by default we leave it as 32 bit.
- `nuql_save_quant_mode_path`: the path to save the quantized model. Quantization nodes  have already been inserted into the graph.
- `nuql_use_buckets`: the switch to turn on the bucket. With bucketing, weights are split into multiple pieces, while the $\alpha$ and $\beta$ are calculated individually for each piece. Therefore, turning on the bucketing can lead to more fine-grained quantization.
- `nuql_bucket_type`: the type of bucketing. Currently two types are supported: [`split`, `channel`]. `split` refers to that the weights of a layer are first concatenated into a long vector, and then cut it into short pieces according to `uql_bucket_size`. The remaining last piece is still regarded as a new piece. After quantization for each piece, the vectors are then folded back to the original shape as the quantized weights. `channel` refers to that weights with shape `[k, k, cin, cout]` in a convolutional layer are cut into `cout` buckets, where each bucket has the size of `k * k * cin`. For weights with shape `[m, n]` in fully connected layers, they are cut into `n` buckets, each of size `m`. In practice, bucketing with type  `channel` can be calculated more quickly comparing to type `split` since there are less buckets and less computation to iterate through all buckets.
- `nuql_bucket_size`: the size of buckets when using bucket type `split`. Generally, smaller bucket size can lead to more fine grained quantization, while more storage are required since full precision statistics ($\alpha$ and $\beta$) of each bucket need to be kept.
- `nuql_quantize_all_layers`: the switch to quantize the first and last layers. The first and last layers of the network are connected directly with the input and output, and are arguably more sensitive to quantization. Keeping them un-quantized can slightly increase the performance, nevertheless, if you want to accelerate the inference speed, all layers are supposed to be quantized.
- `nuql_quant_epoch`: the epochs for fine-tuning a quantized network.
- `nuql_enbl_rl_agent`: the switch to turn on the RL agent as hyper parameter optimizer. Details about the RL agent and its configurations are described below.

### Configure the RL Agent

Similar to uniform quantization, once `nuql_enbl_rl_agent==True` , the RL agent will automatically search for the optimal bit allocation strategy for each layer.  In order to search efficiently, the agent need to be configured properly. While here we list all the configurable hyper parameters for the agent, users can just keep the default value for most parameters, while modify only a few of them if necessary.

| Options                       | Description                                                  |
| :---------------------------- | :----------------------------------------------------------- |
| `nuql_equivalent_bits`       | the number of re-allocated bits that is equivalent to non-uniform quantization without RL agent. Default: `4`. |
| `nuql_nb_rlouts`              | the number of roll outs for training the RL agent. Default: `200`. |
| `nuql_w_bit_min`              | the minimal number of bits for each layer. Default: `2`.     |
| `nuql_w_bit_max`              | the maximal number of bits for each layer. Default: `8`.     |
| `nuql_enbl_rl_global_tune`    | the switch to fine-tune all layers of the network. Default: `True`. |
| `nuql_enbl_rl_layerwise_tune` | the switch to fine-tune the network layer by layer. Default: `False`. |
| `nuql_tune_layerwise_steps`   | the number of steps for layer-wise fine-tuning. Default: `300`. |
| `nuql_tune_global_steps`      | the number of steps for global fine-tuning. Default: `2000`. |
| `nuql_tune_disp_steps`        | the display steps to show the fine-tuning progress. Default: `100`. |
| `nuql_enbl_random_layers`     | the switch to randomly permute layers during RL agent training. Default: `True`. |

Detailed description can be found in [Uniform Quantization](https://pocketflow.github.io/uq_learner/), with the only difference that the prefix is changed to `nuql_`.

## Usage Examples

Again, users should first get the model prepared. Users  can either use the pre-built models in PocketFlow, or develop their customized nets following the model definition in PocketFlow (for example, [resnet_at_cifar10.py](https://github.com/Tencent/PocketFlow/blob/master/nets/resnet_at_cifar10.py)) Once the model is built, the Non-Uniform Quantization Learner can be easily triggered  as follows:

To quantize a ResNet-20 model for CIFAR-10 classification task with 4 bits in the local mode, use:

```bash
# quantize resnet-20 on CIFAR-10
sh ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
--learner=non-uniform \
--nuql_weight_bits=4 \
--nuql_activation_bits=4 \
```

To quantize a ResNet-18 model for ILSVRC_12 classification task with 8 bits in the docker mode with 4 GPUs, and allow to use the channel-wise bucketing, use:

``` bash
# quantize the resnet-18 on ILSVRC-12
sh ./scripts/run_docker.sh nets/resnet_at_ilsvrc12_run.py \
-n=4 \
--learner=non-uniform \
--nuql_weight_bits=8 \
--nuql_activation_bits=8 \
--nuql_use_buckets=True \
--nuql_bucket_type=channel
```

To quantize a MobileNet-v1 model for ILSVRC_12 classification task with 4 bits in the seven mode with 8 GPUs, and allow the RL agent to search for the optimal bit strategy, use:

```bash
# quantize mobilenet-v1 on ILSVRC-12
sh ./scripts/run_seven.sh nets/mobilenet_at_ilsvrc12_run.py \
-n=8 \
--learner=non-uniform \
--nuql_enbl_rl_agent=True \
--nuql_equivalent_bits=4 \
```

## References

Han S, Mao H, and Dally W J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. [arXiv:1510.00149, 2015](https://arxiv.org/abs/1510.00149)
