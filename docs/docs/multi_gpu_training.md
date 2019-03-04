# Multi-GPU Training

Due to the high computational complexity, it often takes hours or even days to fully train deep learning models using a single GPU.
In PocketFlow, we adopt multi-GPU training to speed-up this time-consuming training process.
Our implementation is compatible with:

* [Horovod](https://github.com/uber/horovod): a distributed training framework for TensorFlow, Keras, and PyTorch.
* TF-Plus: an optimized framework for TensorFlow-based distributed training (only available within Tencent).

We have provide a wrapper class, `MultiGpuWrapper`, to seamlessly switch between the above two frameworks.
It will sequentially check whether Horovod and TF-Plus can be used, and use the first available one as the underlying framework for multi-GPU training.

The main reason that using Horovod or TF-Plus instead TensorFlow's original distributed training routine is that these frameworks provide many easy-to-use APIs and require far less code changes to change from single-GPU to multi-GPU training, as we shall see later.

## From Single-GPU to Multi-GPU

To extend a single-GPU based training script to the multi-GPU scenario, at most 7 steps are needed:

* Import the Horovod or TF-Plus module.

``` Python
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
```

* Initialize the multi-GPU training framework, as early as possible.

``` Python
mgw.init()
```

* For each worker, create a session with a distinct GPU device.

``` Python
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(mgw.local_rank())
sess = tf.Session(config=config)
```

* (Optional) Let each worker use a distinct subset of training data.

``` Python
filenames = tf.data.Dataset.list_files(file_pattern, shuffle=True)
filenames = filenames.shard(mgw.size(), mgw.rank())
```

* Wrapper the optimizer for distributed gradient communication.

``` Python
optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate)
optimizer = mgw.DistributedOptimizer(optimizer)
train_op = optimizer.minimize(loss)
```

* Synchronize master's parameters to all the other workers.

``` Python
bcast_op = mgw.broadcast_global_variables(0)
sess.run(tf.global_variables_initializer())
sess.run(bcast_op)
```

* (Optional) Save checkpoint files at the master node periodically.

``` Python
if mgw.rank() == 0:
  saver.save(sess, save_path, global_step)
```

## Usage Example

Here, we provide a code snippet to demonstrate how to use multi-GPU training to speed-up training.
Please note that many implementation details are omitted for clarity.

``` Python
import tensorflow as tf
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

# initialization
mgw.init()

# create the training graph
with tf.Graph().as_default():
  # create a TensorFlow session
  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(mgw.local_rank())
  sess = tf.Session(config=config)

  # use tf.data.Dataset() to traverse images and labels
  filenames = tf.data.Dataset.list_files(file_pattern, shuffle=True)
  filenames = filenames.shard(mgw.size(), mgw.rank())
  images, labels = get_images_n_labels(filenames)

  # define the network and its loss function
  logits = forward_pass(images)
  loss = calc_loss(labels, logits)

  # create an optimizer and setup training-related operations
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate)
  optimizer = mgw.DistributedOptimizer(optimizer)
  train_op = optimizer.minimize(loss, global_step=global_step)
  bcast_op = mgw.broadcast_global_variables(0)

# multi-GPU training
sess.run(tf.global_variables_initializer())
sess.run(bcast_op)
for idx_iter in range(nb_iters):
  sess.run(train_op)
  if mgw.rank() == 0 and (idx_iter + 1) % save_step == 0:
    saver.save(sess, save_path, global_step)
```
