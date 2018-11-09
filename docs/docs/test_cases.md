# Test Cases

This document contains various test cases to cover different combinations of learners and hyper-parameter settings. Any merge request to the master branch should be able to pass all the test cases to be approved.

## Full-Precision

``` bash
# local mode
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --enbl_dst
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --data_disk hdfs
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --data_disk hdfs \
    --enbl_dst

# seven mode
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py \
    --enbl_dst
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py \
    --data_disk hdfs
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py \
    --data_disk hdfs \
    --enbl_dst

# docker mode
$ ./scripts/run_docker.sh nets/lenet_at_cifar10_run.py
$ ./scripts/run_docker.sh nets/lenet_at_cifar10_run.py \
    --enbl_dst
$ ./scripts/run_docker.sh nets/resnet_at_cifar10_run.py
$ ./scripts/run_docker.sh nets/resnet_at_cifar10_run.py \
    --enbl_dst
```

## Channel Pruning

``` bash
# uniform preserve ratios for all layers
$ ./scripts/run_seven.sh nets/resnet_at_cifar10_run.py \
    --learner channel \
    --cp_prune_option uniform \
    --cp_uniform_preserve_ratio 0.5

# auto-tuned preserve ratios for each layer
$ ./scripts/run_seven.sh nets/resnet_at_cifar10_run.py \
    --cp_learner channel \
    --cp_prune_option auto \
    --cp_preserve_ratio 0.3
```

## Discrimination-aware Channel Pruning

``` bash
# no network distillation
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner dis-chn-pruned \
    --dcp_nb_stages 3 \
    --data_disk hdfs

# network distillation
$ ./scripts/run_seven.sh nets/mobilenet_at_ilsvrc12_run.py \
    --learner dis-chn-pruned \
    --enbl_dst \
    --dcp_nb_stages 4
```

## Weight Sparsification

``` bash
# uniform pruning ratios for all layers
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner weight-sparse \
    --ws_prune_ratio_prtl uniform \
    --data_disk hdfs

# optimal pruning ratios for each layer
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner weight-sparse \
    --ws_prune_ratio_prtl optimal \
    --data_disk hdfs

# heurist pruning ratios for each layer
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py \
    --learner weight-sparse \
    --ws_prune_ratio_prtl heurist

# optimal pruning ratios for each layer
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py \
    --learner weight-sparse \
    --ws_prune_ratio_prtl optimal
```

## Uniform Quantization

``` bash
# channel-based bucketing
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner uniform \
    --uql_use_buckets \
    --uql_bucket_type channel \
    --data_disk hdfs

# split-based bucketing
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner uniform \
    --uql_use_buckets \
    --uql_bucket_type split \
    --data_disk hdfs

# channel-based bucketing + RL
$ ./scripts/run_seven.sh nets/mobilenet_at_ilsvrc12_run.py -n=2 \
    --learner uniform \
    --uql_enbl_rl_agent \
    --uql_use_buckets \
    --uql_bucket_type channel

# split-based bucketing + RL
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py -n=2 \
    --learner uniform \
    --uql_enbl_rl_agent \
    --uql_use_buckets \
    --uql_bucket_type split
```

## Non-uniform Quantization

``` bash
# channel-based bucketing + RL + optimize clusters
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner non-uniform \
    --nuql_enbl_rl_agent \
    --nuql_use_buckets \
    --nuql_bucket_type channel \
    --nuql_opt_mode clusters \
    --data_disk hdfs

# split-based bucketing + RL + optimize weights
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py \
    --learner non-uniform \
    --nuql_enbl_rl_agent \
    --nuql_use_buckets \
    --nuql_bucket_type split \
    --nuql_opt_mode weights \
    --data_disk hdfs

# channel-based bucketing + RL + optimize weights
$ ./scripts/run_seven.sh nets/mobilenet_at_ilsvrc12_run.py -n=2 \
    --learner non-uniform \
    --nuql_enbl_rl_agent \
    --nuql_use_buckets \
    --nuql_bucket_type channel \
    --nuql_opt_mode weights

# split-based bucketing + RL + optimize clusters
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py -n=2 \
    --learner non-uniform \
    --nuql_enbl_rl_agent \
    --nuql_use_buckets \
    --nuql_bucket_type split \
    --nuql_opt_mode clusters
```
