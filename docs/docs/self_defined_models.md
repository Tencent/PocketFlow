# Self-defined Models

Self-defined models (and data sets) can be incorporated into PocketFlow by implementing a new `ModelHelper` class. The `ModelHelper` class includes the definition of data input pipeline as well as the network's forward pass and loss function. With the self-defined `ModelHelper`, the network can be either trained without any constraints using `FullPrecLearner`, or trained with certain model compression algorithms using other learners, *e.g.* `ChannelPrunedLearner` for channel pruning or `UniformQuantTFLearner` for uniform quantization. In this tutorial, we will define a 4-layer convolutional neural network (2 conv. layers + 2 dense layers) for image classification on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) data set under the PocketFlow framework. Afterwards, we shall demonstrate how to train this self-defined model with different model compression components.

## The Essentials

To use self-defined models and data sets in PocketFlow, we need to provide the following two items in advance to describe the overall training workflow:

* **Data Input Pipeline**: this tells PocketFlow how to parse features and ground-truth labels from data files.
* **Network Definition**: this tells PocketFlow how to compute the network's predictions and loss function's value.

The `ModelHelper` class, which is a sub-class of the abstract base class `AbstractModelHelper`, is designed to provide such definitions. In PocketFlow, we have offered several `ModelHelper`  classes to describe different combinations of data sets and model architectures. To use self-defined models, a new `ModelHelper` class should be implemented. Besides, we need an execution script to call this newly defined `ModelHelper` class.

P.S.: You can find the full code used in this tutorial under the "./examples" directory.

### Data Input Pipeline

To start with, we need to tell PocketFlow how data files should be parsed. Here, we define a class named `FMnistDataset` to create iterators over the Fashion-MNIST training and test subsets. Every time the iterator is called, it will return a mini-batch of images and corresponding ground-truth labels.

Below is the full implementation of `FMnistDataset` class (this should be placed under the "./datasets" directory, named as "fmnist_dataset.py"):

``` Python
import os
import gzip
import numpy as np
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 10, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 60000, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 5000, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 10000, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 100, 'batch size for evaluation')

# Fashion-MNIST specifications
IMAGE_HEI = 28
IMAGE_WID = 28
IMAGE_CHN = 1

def load_mnist(image_file, label_file):
  """Load images and labels from *.gz files.

  This function is modified from utils/mnist_reader.py in the Fashion-MNIST repo.

  Args:
  * image_file: file path to images
  * label_file: file path to labels

  Returns:
  * images: np.array of the image data
  * labels: np.array of the label data
  """

  with gzip.open(label_file, 'rb') as i_file:
    labels = np.frombuffer(i_file.read(), dtype=np.uint8, offset=8)
  with gzip.open(image_file, 'rb') as i_file:
    images = np.frombuffer(i_file.read(), dtype=np.uint8, offset=16)
    image_size = IMAGE_HEI * IMAGE_WID * IMAGE_CHN
    assert images.size == image_size * len(labels)
    images = images.reshape(len(labels), image_size)

  return images, labels

def parse_fn(image, label, is_train):
  """Parse an (image, label) pair and apply data augmentation if needed.

  Args:
  * image: image tensor
  * label: label tensor
  * is_train: whether data augmentation should be applied

  Returns:
  * image: image tensor
  * label: one-hot label tensor
  """

  # data parsing
  label = tf.one_hot(tf.reshape(label, []), FLAGS.nb_classes)
  image = tf.cast(tf.reshape(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN]), tf.float32)
  image = tf.image.per_image_standardization(image)

  # data augmentation
  if is_train:
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEI + 8, IMAGE_WID + 8)
    image = tf.random_crop(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN])
    image = tf.image.random_flip_left_right(image)

  return image, label

class FMnistDataset(AbstractDataset):
  '''Fashion-MNIST dataset.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # initialize the base class
    super(FMnistDataset, self).__init__(is_train)

    # choose local files or HDFS files w.r.t. FLAGS.data_disk
    if FLAGS.data_disk == 'local':
      assert FLAGS.data_dir_local is not None, '<FLAGS.data_dir_local> must not be None'
      data_dir = FLAGS.data_dir_local
    elif FLAGS.data_disk == 'hdfs':
      assert FLAGS.data_hdfs_host is not None and FLAGS.data_dir_hdfs is not None, \
        'both <FLAGS.data_hdfs_host> and <FLAGS.data_dir_hdfs> must not be None'
      data_dir = FLAGS.data_hdfs_host + FLAGS.data_dir_hdfs
    else:
      raise ValueError('unrecognized data disk: ' + FLAGS.data_disk)

    # setup paths to image & label files, and read in images & labels
    if is_train:
      self.batch_size = FLAGS.batch_size
      image_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
      label_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    else:
      self.batch_size = FLAGS.batch_size_eval
      image_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
      label_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    self.images, self.labels = load_mnist(image_file, label_file)
    self.parse_fn = lambda x, y: parse_fn(x, y, is_train)

  def build(self, enbl_trn_val_split=False):
    """Build iterator(s) for tf.data.Dataset() object.

    Args:
    * enbl_trn_val_split: whether to split into training & validation subsets

    Returns:
    * iterator_trn: iterator for the training subset
    * iterator_val: iterator for the validation subset
      OR
    * iterator: iterator for the chosen subset (training OR testing)
    """

    # create a tf.data.Dataset() object from NumPy arrays
    dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
    dataset = dataset.map(self.parse_fn, num_parallel_calls=FLAGS.nb_threads)

    # create iterators for training & validation subsets separately
    if self.is_train and enbl_trn_val_split:
      iterator_val = self.__make_iterator(dataset.take(FLAGS.nb_smpls_val))
      iterator_trn = self.__make_iterator(dataset.skip(FLAGS.nb_smpls_val))
      return iterator_trn, iterator_val

    return self.__make_iterator(dataset)

  def __make_iterator(self, dataset):
    """Make an iterator from tf.data.Dataset.

    Args:
    * dataset: tf.data.Dataset object

    Returns:
    * iterator: iterator for the dataset
    """

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size))
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator
```

When creating an object of `FMnistDataset` class, an extra argument named `is_train` should be provided to toggle between the training and test subsets. The data files can be either store on the local machine or the HDFS cluster, and the directory path is specified in the path configuration file, *e.g.*:

``` plain
data_dir_local_fmnist = /home/user_name/datasets/Fashion-MNIST
```

The constructor function loads images and labels from *.gz files, each stored in a NumPy array. The `build` function is then used to create a TensorFlow's data set iterator from these two NumPy arrays. Particularly, if both `enbl_trn_val_split` and `is_train` are True, then the original training subset will be divided into two parts, one for model training and the other for validation.

### Network Definition

Now we implement a new `ModelHelper` class to utilize the above data input pipeline to define the network's training workflow. Below is the full implementation of `ModelHelper` class (this should be placed under the "./nets" directory, named as "convnet_at_fmnist.py"):

``` Python
import tensorflow as tf

from nets.abstract_model_helper import AbstractModelHelper
from datasets.fmnist_dataset import FMnistDataset
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\' ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')

def forward_fn(inputs, data_format):
  """Forward pass function.

  Args:
  * inputs: inputs to the network's forward pass
  * data_format: data format ('channels_last' OR 'channels_first')

  Returns:
  * inputs: outputs from the network's forward pass
  """

  # transpose the image tensor if needed
  if data_format == 'channel_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  # conv1
  inputs = tf.layers.conv2d(inputs, 32, [5, 5], padding='same',
                            data_format=data_format, activation=tf.nn.relu, name='conv1')
  inputs = tf.layers.max_pooling2d(inputs, [2, 2], 2, data_format=data_format, name='pool1')

  # conv2
  inputs = tf.layers.conv2d(inputs, 64, [5, 5], padding='same',
                            data_format=data_format, activation=tf.nn.relu, name='conv2')
  inputs = tf.layers.max_pooling2d(inputs, [2, 2], 2, data_format=data_format, name='pool2')

  # fc3
  inputs = tf.layers.flatten(inputs, name='flatten')
  inputs = tf.layers.dense(inputs, 1024, activation=tf.nn.relu, name='fc3')

  # fc4
  inputs = tf.layers.dense(inputs, FLAGS.nb_classes, name='fc4')
  inputs = tf.nn.softmax(inputs, name='softmax')

  return inputs

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a ConvNet model for the Fashion-MNIST dataset."""

  def __init__(self):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__()

    # initialize training & evaluation subsets
    self.dataset_train = FMnistDataset(is_train=True)
    self.dataset_eval = FMnistDataset(is_train=False)

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build(enbl_trn_val_split)

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs, data_format='channels_last'):
    """Forward computation at training."""

    return forward_fn(inputs, data_format)

  def forward_eval(self, inputs, data_format='channels_last'):
    """Forward computation at evaluation."""

    return forward_fn(inputs, data_format)

  def calc_loss(self, labels, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    loss = tf.losses.softmax_cross_entropy(labels, outputs)
    loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])
    accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), tf.float32))
    metrics = {'accuracy': accuracy}

    return loss, metrics

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    nb_epochs = 160
    idxs_epoch = [40, 80, 120]
    decay_rates = [1.0, 0.1, 0.01, 0.001]
    batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
    nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

    return lrn_rate, nb_iters

  @property
  def model_name(self):
    """Model's name."""

    return 'convnet'

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'fmnist'
```

In the `build_dataset_train` and `build_dataset_eval` functions, we adopt the previously introduced `FMnistDataset` class to define the data input pipeline. The network forward-pass computation is defined in the `forward_train` and `forward_eval` functions, which corresponds to the training and evaluation graph, respectively. The training graph is slightly different from evaluation graph, such as operations related to the batch normalization layers. The `calc_loss` function calculates the loss function's value and extra evaluation metrics, *e.g.* classification accuracy. Finally, the `setup_lrn_rate` function defines the learning rate schedule, as well as how many training iterations are need.

### Execution Script

Besides the self-defined `ModelHelper` class, we still need an execution script to pass it to the corresponding model compression component to start the training process. Below is the full implementation (this should be placed under the "./nets" directory, named as "convnet_at_fmnist_run.py"):

``` Python
import traceback
import tensorflow as tf

from nets.convnet_at_fmnist import ModelHelper
from learners.learner_utils import create_learner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_boolean('enbl_multi_gpu', False, 'enable multi-GPU training')
tf.app.flags.DEFINE_string('learner', 'full-prec', 'learner\'s name')
tf.app.flags.DEFINE_string('exec_mode', 'train', 'execution mode: train / eval')
tf.app.flags.DEFINE_boolean('debug', False, 'debugging information')

def main(unused_argv):
  """Main entry."""

  try:
    # setup the TF logging routine
    if FLAGS.debug:
      tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
      tf.logging.set_verbosity(tf.logging.INFO)
    sm_writer = tf.summary.FileWriter(FLAGS.log_dir)

    # display FLAGS's values
    tf.logging.info('FLAGS:')
    for key, value in FLAGS.flag_values_dict().items():
      tf.logging.info('{}: {}'.format(key, value))

    # build the model helper & learner
    model_helper = ModelHelper()
    learner = create_learner(sm_writer, model_helper)

    # execute the learner
    if FLAGS.exec_mode == 'train':
      learner.train()
    elif FLAGS.exec_mode == 'eval':
      learner.download_model()
      learner.evaluate()
    else:
      raise ValueError('unrecognized execution mode: ' + FLAGS.exec_mode)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
```

## Network Training with PocketFlow

To train the self-defined model without any constraint, use `FullPrecLearner`:

``` bash
$ ./scripts/run_local.sh nets/convnet_at_fmnist_run.py \
    --learner full-prec
```

To train the self-defined model with the uniform quantization constraint, use `UniformQuantTFLearner`:

``` bash
$ ./scripts/run_local.sh nets/convnet_at_fmnist_run.py \
    --learner uniform-tf
```
