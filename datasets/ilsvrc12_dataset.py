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
"""ILSVRC-12 dataset."""

import os
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset
from utils.external.imagenet_preprocessing import preprocess_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 1001, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 1281167, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 10000, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 50000, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 100, 'batch size for evaluation')

# ILSVRC-12 specifications
IMAGE_HEI = 224
IMAGE_WID = 224
IMAGE_CHN = 3

def parse_example_proto(example_serialized):
  """Parse image buffer, label, and bounding box from the serialized data.

  Args:
  * example_serialized: serialized example data

  Returns:
  * image_buffer: image buffer label
  * label: label tensor (not one-hot)
  * bbox: bounding box tensor
  """

  # parse features from the serialized data
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
  }
  bbox_keys = ['image/object/bbox/' + x for x in ['xmin', 'ymin', 'xmax', 'ymax']]
  feature_map.update({key: tf.VarLenFeature(dtype=tf.float32) for key in bbox_keys})
  features = tf.parse_single_example(example_serialized, feature_map)

  # obtain the label and bounding boxes
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox

def parse_fn(example_serialized, is_train):
  """Parse image & labels from the serialized data.

  Args:
  * example_serialized: serialized example data
  * is_train: whether data augmentation should be applied

  Returns:
  * image: image tensor
  * label: one-hot label tensor
  """

  image_buffer, label, bbox = parse_example_proto(example_serialized)
  image = preprocess_image(
    image_buffer=image_buffer, bbox=bbox, output_height=IMAGE_HEI,
    output_width=IMAGE_WID, num_channels=IMAGE_CHN, is_training=is_train)
  label = tf.one_hot(tf.reshape(label, []), FLAGS.nb_classes)

  return image, label

class Ilsvrc12Dataset(AbstractDataset):
  '''ILSVRC-12 dataset.'''

  def __init__(self, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # initialize the base class
    super(Ilsvrc12Dataset, self).__init__(is_train)

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
      self.file_pattern = os.path.join(data_dir, 'train-*-of-*')
      self.batch_size = FLAGS.batch_size
    else:
      self.file_pattern = os.path.join(data_dir, 'validation-*-of-*')
      self.batch_size = FLAGS.batch_size_eval
    self.dataset_fn = tf.data.TFRecordDataset
    self.parse_fn = lambda x: parse_fn(x, is_train=is_train)
