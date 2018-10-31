# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Abstract class for datasets."""

from abc import ABC
import tensorflow as tf

from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_disk', 'local', 'data disk\'s location (\'local\' | \'hdfs\')')
tf.app.flags.DEFINE_string('data_hdfs_host', None, 'HDFS host for data files')
tf.app.flags.DEFINE_string('data_dir_local', None, 'data directory - local')
tf.app.flags.DEFINE_string('data_dir_hdfs', None, 'data directory - HDFS')
tf.app.flags.DEFINE_integer('cycle_length', 4, '# of datasets to interleave from in parallel')
tf.app.flags.DEFINE_integer('nb_threads', 8, '# of threads for preprocessing the dataset')
tf.app.flags.DEFINE_integer('buffer_size', 1024, '# of elements to be buffered when prefetching')
tf.app.flags.DEFINE_integer('prefetch_size', 8, '# of mini-batches to be buffered when prefetching')

class AbstractDataset(ABC):
  '''Abstract class for datasets.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # following attributes must be initialized by each sub-class
    self.file_pattern = None
    self.dataset_fn = None
    self.parse_fn = None
    self.batch_size = None

    # determine whether data sharding is enabled
    self.is_train = is_train
    self.enbl_shard = (is_train and FLAGS.enbl_multi_gpu)  # shard files for multi-GPU training

  def build(self, enbl_trn_val_split=False):
    '''Build iterator(s) for tf.data.Dataset() object.

    Args:
    * enbl_trn_val_split: whether to split into training & validation subsets

    Returns:
    * iterator_trn: iterator for the training subset
    * iterator_val: iterator for the validation subset
      OR
    * iterator: iterator for the chosen subset (training OR testing)

    Example:
      # build iterator(s)
      dataset = xxxxDataset(is_train=True)  # TF operations are not created
      iterator = dataset.build()            # TF operations are created
          OR
      iterator_trn, iterator_val = dataset.build(enbl_trn_val_split=True)  # for dataset-train only

      # use the iterator to obtain a mini-batch of images & labels
      images, labels = iterator.get_next()
    '''

    # obtain list of data files' names
    filenames = tf.data.Dataset.list_files(self.file_pattern, shuffle=True)
    if self.enbl_shard:
      filenames = filenames.shard(mgw.size(), mgw.rank())

    # create a tf.data.Dataset from list of files
    dataset = filenames.apply(
      tf.contrib.data.parallel_interleave(self.dataset_fn, cycle_length=FLAGS.cycle_length))
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
