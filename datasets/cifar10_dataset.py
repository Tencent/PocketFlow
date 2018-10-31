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
"""CIFAR-10 dataset."""

import os
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 10, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 50000, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 5000, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 10000, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 100, 'batch size for evaluation')

# CIFAR-10 specifications
LABEL_BYTES = 1
IMAGE_HEI = 32
IMAGE_WID = 32
IMAGE_CHN = 3
IMAGE_BYTES = IMAGE_CHN * IMAGE_HEI * IMAGE_WID
RECORD_BYTES = LABEL_BYTES + IMAGE_BYTES
IMAGE_AVE = tf.constant([[[125.3, 123.0, 113.9]]], dtype=tf.float32)
IMAGE_STD = tf.constant([[[63.0, 62.1, 66.7]]], dtype=tf.float32)

def parse_fn(example_serialized, is_train):
  """Parse image & labels from the serialized data.

  Args:
  * example_serialized: serialized example data
  * is_train: whether data augmentation should be applied

  Returns:
  * image: image tensor
  * label: one-hot label tensor
  """

  # data parsing
  record = tf.decode_raw(example_serialized, tf.uint8)
  label = tf.slice(record, [0], [LABEL_BYTES])
  label = tf.one_hot(tf.reshape(label, []), FLAGS.nb_classes)
  image = tf.slice(record, [LABEL_BYTES], [IMAGE_BYTES])
  image = tf.reshape(image, [IMAGE_CHN, IMAGE_HEI, IMAGE_WID])
  image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)
  image = (image - IMAGE_AVE) / IMAGE_STD

  # data augmentation
  if is_train:
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_HEI + 8, IMAGE_WID + 8)
    image = tf.random_crop(image, [IMAGE_HEI, IMAGE_WID, IMAGE_CHN])
    image = tf.image.random_flip_left_right(image)

  return image, label

class Cifar10Dataset(AbstractDataset):
  '''CIFAR-10 dataset.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # initialize the base class
    super(Cifar10Dataset, self).__init__(is_train)

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

    # configure file patterns & function handlers
    if is_train:
      self.file_pattern = os.path.join(data_dir, 'data_batch_*.bin')
      self.batch_size = FLAGS.batch_size
    else:
      self.file_pattern = os.path.join(data_dir, 'test_batch.bin')
      self.batch_size = FLAGS.batch_size_eval
    self.dataset_fn = lambda x: tf.data.FixedLengthRecordDataset(x, RECORD_BYTES)
    self.parse_fn = lambda x: parse_fn(x, is_train=is_train)
