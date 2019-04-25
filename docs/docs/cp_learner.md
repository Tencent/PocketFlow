# Channel Pruning

## Introduction

Channel pruning is a kind of structural model compression approach which can not only compress the model size, but accelerate the inference speed directly. PocketFlow uses the channel pruning algorithm proposed in (He et al., 2017) to pruning each channel of convolution layers with a certain ratio, and for details please refer to the [channel pruning paper](https://arxiv.org/abs/1707.06168). For better performance and more robust, we modify some parts of the algorithm to achieve better result.

In order to achieve a better performance, PocketFlow can take advantages of reinforcement learning to search a better compression ratio (He et al., 2018). User can also use the distilling (Hinton et al., 2015) and group tuning function to improve the accuracy after compression. Group tuning means setting a certain number of layers as group and then pruning and fine-tuning (or re-training) each group sequentially. For example, we can set each 3 layers as a group and then prune the first 3 layers. After that, we fine-tune (or re-train) the whole model and prune the next 3 layers and so on. Distilling and group tuning are experimentally proved as effective approaches to achieve higher accuracy at a certain compression ratio in most situations.

## Pruning Option

The code of channel pruning are located at directory `./learners/channel_pruning`. To use channel pruning. users can set `--learners` to `channel`. The Channel pruning supports 3 kinds of pruning setup by `cp_prune_option` option.

### Uniform Channel Pruning

One is the uniform layer pruning, which means the user can set each convolution layer pruned with an uniform pruning ratio by  `--cp_prune_option=uniform` and set the ratio (eg. making the ratio 0.5) by `--cp_uniform_preserve_ratio=0.5`. Note that for a layer, if both of pruning ratio of the layer and its previous layer are 0.5, the real preserved FLOPs are 1/4 of original FLOPs. Because channel pruning only prune the c_out channels of the convolution and c_in channels of the next convolution, if both c_in and c_out channels are pruned by 0.5, it will preserve only 1/4 of original computation cost. For a layer by layer convolution networks without residual blocks, if the user set `cp_uniform_preserve_ratio` to `0.5`, the whole model will be the 0.25 computation of the original model. However for the residual networks, some convolutions can only prune their c_in or c_out channels, which means the total preserved computation ratio may be much greater than 0.25.

**Example:**

``` bash
$ ./scripts/run_seven.sh nets/resnet_at_cifar10_run.py \
    --learner channel \
    --batch_size_eval 64 \
    --cp_uniform_preserve_ratio 0.5 \
    --cp_prune_option uniform \
    --resnet_size 20
```

### List Channel Pruning

Another pruning option is pruning the corresponding layer with ratios listed in a named `ratio.list` file, the file name of which can be set by `--cp_prune_list_file` option. the ratio value must be separated by a comma. User can set `--cp_prune_option=list` to prune the model by list ratios.

**Example:**
Add list `1.0, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 1.0, 0.25, 1.0, 0.25, 0.21875, 0.21875, 0.21875, 1.0, 0.5625, 1.0, 0.546875, 0.546875, 0.546875, 1` in `./ratio.list`

``` bash
$ ./scripts/run_seven.sh nets/resnet_at_cifar10_run.py \
    --learner channel \
    --batch_size_eval 64 \
    --cp_prune_option list \
    --cp_prune_list_file ./ratio.list \
    --resnet_size 20
```

### Automatic Channel Pruning

The last one pruning option is searching better pruning ratios by reinforcement learning and you only need to give a value which represents what the ratio of total FLOPs/Computation you wants the compressed model preserve. You can set `--cp_prune_option=auto` and set a preserve ratio number such as `--cp_preserve_ratio=0.5`.  User can also use `cp_nb_rlouts_min` to control reinforcement learning warm up iterations, which means the RL agent start to learn after the iterations, the default value is `50`. User can also use `cp_nb_rlouts` to control the total iteration RL agent to search, the default value is `200`. If the user want to control other parameters of the agents, please refer to the reinforcement component page.

**Example:**

``` bash
$ ./scripts/run_seven.sh nets/resnet_at_cifar10_run.py \
    --learner channel \
    --batch_size_eval 64 \
    --cp_preserve_ratio 0.5 \
    --cp_prune_option auto \
    --resnet_size 20
```

## Channel pruning parameters

The implementation of the channel pruning use Lasso algorithm to do channel selection and linear regression to do feature map reconstruction. During these two phases, sampling is done on the feature map to reduce computation cost. The users can use `--cp_nb_points_per_layer` to set how many sampling points on each layer are taken, the default value is `10`. For some dataset, if the images contain too many zero pixels (eg. black color), the value should be greater. The users can also set using how many batches to do channel selection and feature reconstruction by `cp_nb_batches`, the default value is `60`. Small value of  `cp_nb_batches` may cause over-fitting and large value may slow down the solving speed, so a good value depends on the nets and dataset. For more practical usage, user may consider make the channel number of each layer is the quadruple for fast inference of mobile devices. In this case, user can set `--cp_quadruple` to `True` to make the compressed model have a quadruple number of channels.

## Distilling

Distilling is an effective approach to improve the final accuracy of compressed model with PocketFlow in most situations of classification. User can set `--enbl_dst=True` to enable distilling.

## Group Tuning

As introduced above, group tuning was proposed by the PocketFlow team and finding it is very useful to improve the performance of model compression. In PocketFlow, users can set `--cp_finetune=True` to enable group fine-tuning and set the group number by `--cp_list_group`, the default value is `1000`. There is a trade-off between the small value and large value, because if the value is `1`, PocketFlow will prune convolution and fine-tune/re-train by each layer, which may have better effect but be more time-consuming. If we set the value large, the function will be less effective. User can also set the number of iterations to fine-tune by setting `cp_nb_iters_ft_ratio` which mean the ratio the total iterations to be used in fine-tuning. The learning rate of fine-tuning can be set by `cp_lrn_rate_ft`.
