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
"""Pascal VOC dataset."""

import os
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 300, 'output image size')
tf.app.flags.DEFINE_integer('image_size_eval', 300, 'output image size for evaluation')
tf.app.flags.DEFINE_integer('nb_bboxs_max', 100, 'maximal # of bounding boxes per image')
tf.app.flags.DEFINE_integer('nb_classes', 21, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 22136, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 500, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 4952, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 1, 'batch size for evaluation')

# Pascal VOC specifications
IMAGE_CHN = 3

def parse_example_proto(example_serialized):
  """Parse the unserialized feature data from the serialized data.

  Args:
  * example_serialized: serialized example data

  Returns:
  * features: unserialized feature data
  """

  # parse features from the serialized data
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value='jpeg'),
    'image/filename': tf.FixedLenFeature((), dtype=tf.string, default_value=''),
    'image/height': tf.FixedLenFeature([1], dtype=tf.int64),
    'image/width': tf.FixedLenFeature([1], dtype=tf.int64),
    'image/channels': tf.FixedLenFeature([1], dtype=tf.int64),
    'image/shape': tf.FixedLenFeature([3], dtype=tf.int64),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
  }
  features = tf.parse_single_example(example_serialized, feature_map)

  return features

def pack_annotations(bboxes, labels, difficults=None, truncateds=None):
  """Pack all the annotations into one tensor.

  Args:
  * bboxes: list of bounding box coordinates (N x 4)
  * labels: list of category labels (N)
  * difficults: list of difficulty flags (N)
  * truncateds: list of truncation flags (N)

  Returns:
  * objects: one tensor with all the annotations packed together (FLAGS.nb_bboxs_max x 8)
  """

  # pack <bboxes> & <labels> with a leading <flags>
  bboxes = tf.cast(bboxes, tf.float32)
  labels = tf.cast(tf.expand_dims(labels, 1), tf.float32)
  flags = tf.ones(tf.shape(labels))
  objects = tf.concat([flags, bboxes, labels], axis=1)

  # pack <difficults> & <truncateds> if supplied
  if difficults is not None and truncateds is not None:
    difficults = tf.cast(tf.expand_dims(difficults, 1), tf.float32)
    truncateds = tf.cast(tf.expand_dims(truncateds, 1), tf.float32)
    objects = tf.concat([objects, difficults, truncateds], axis=1)

  # pad to fixed number of bounding boxes
  pad_size = FLAGS.nb_bboxs_max - tf.shape(objects)[0]
  objects = tf.pad(objects, [[0, pad_size], [0, 0]])

  return objects

def parse_fn(example_serialized, preprocess_fn, is_train):
  """Parse image & objects from the serialized data.

  Args:
  * example_serialized: serialized example data
  * preprocess_fn: preprocessing function
  * is_train: whether to construct the training subset

  Returns:
  * image: image tensor
  * objects: one tensor with all the annotations packed together
  """

  # unserialize the example proto
  features = parse_example_proto(example_serialized)

  # obtain the image data
  image_raw = tf.image.decode_jpeg(features['image/encoded'], channels=IMAGE_CHN)
  filename = features['image/filename']
  shape = features['image/shape']

  # obtain bounding boxes' coordinates
  # Note that we impose an ordering of (y, x) just to make life difficult.
  xmins = tf.expand_dims(features['image/object/bbox/xmin'].values, 1)
  ymins = tf.expand_dims(features['image/object/bbox/ymin'].values, 1)
  xmaxs = tf.expand_dims(features['image/object/bbox/xmax'].values, 1)
  ymaxs = tf.expand_dims(features['image/object/bbox/ymax'].values, 1)
  bboxes_raw = tf.concat([ymins, xmins, ymaxs, xmaxs], axis=1)  # N x 4

  # obtain other annotation data
  labels_raw = tf.cast(features['image/object/bbox/label'].values, tf.int64)
  difficults = tf.cast(features['image/object/bbox/difficult'].values, tf.int64)
  truncateds = tf.cast(features['image/object/bbox/truncated'].values, tf.int64)

  # filter out difficult objects
  if is_train:
    # if all is difficult, then keep the first one; otherwise, use all the non-difficult objects
    mask = tf.cond(
      tf.count_nonzero(difficults, dtype=tf.int32) < tf.shape(difficults)[0],
      lambda: difficults < tf.ones_like(difficults),
      lambda: tf.one_hot(0, tf.shape(difficults)[0], on_value=True, off_value=False, dtype=tf.bool))
    labels_raw = tf.boolean_mask(labels_raw, mask)
    bboxes_raw = tf.boolean_mask(bboxes_raw, mask)

  # pre-process image, labels, and bboxes
  data_format = 'channels_last'  # use the channel-last ordering by default
  if is_train:
    out_shape = [FLAGS.image_size, FLAGS.image_size]
    image, labels, bboxes = preprocess_fn(
      image_raw, labels_raw, bboxes_raw, out_shape,
      is_training=True, data_format=data_format, output_rgb=False)
  else:
    out_shape = [FLAGS.image_size_eval, FLAGS.image_size_eval]
    image = preprocess_fn(
      image_raw, labels_raw, bboxes_raw, out_shape,
      is_training=False, data_format=data_format, output_rgb=False)
    labels, bboxes = labels_raw, bboxes_raw

  # pack all the annotations into one tensor
  image_info = {'image': image, 'filename': filename, 'shape': shape}
  objects = pack_annotations(bboxes, labels)

  return image_info, objects

class PascalVocDataset(AbstractDataset):
  """Pascal VOC dataset."""

  def __init__(self, preprocess_fn, is_train):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # initialize the base class
    super(PascalVocDataset, self).__init__(is_train)

    # choose local files or HDFS files w.r.t. FLAGS.data_disk
    if FLAGS.data_disk == 'local':
      assert FLAGS.data_dir_local is not None, '<FLAGS.data_dir_local> must not be None'
      self.data_dir = FLAGS.data_dir_local
    elif FLAGS.data_disk == 'hdfs':
      assert FLAGS.data_hdfs_host is not None and FLAGS.data_dir_hdfs is not None, \
        'both <FLAGS.data_hdfs_host> and <FLAGS.data_dir_hdfs> must not be None'
      self.data_dir = FLAGS.data_hdfs_host + FLAGS.data_dir_hdfs
    else:
      raise ValueError('unrecognized data disk: ' + FLAGS.data_disk)

    # configure file patterns & function handlers
    if is_train:
      self.file_pattern = os.path.join(self.data_dir, '*train*')
      self.batch_size = FLAGS.batch_size
    else:
      self.file_pattern = os.path.join(self.data_dir, '*val*')
      self.batch_size = FLAGS.batch_size_eval
    self.dataset_fn = tf.data.TFRecordDataset
    self.parse_fn = lambda x: parse_fn(x, preprocess_fn=preprocess_fn, is_train=is_train)
