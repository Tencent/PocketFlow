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
"""Convert the data format from channels_last (NHWC) to channels_first (NCHW), or vice versa."""

import os
import traceback
import tensorflow as tf

# NOTE: un-comment the corresponding <ModelHelper> before conversion
#from nets.lenet_at_cifar10 import ModelHelper
#from nets.resnet_at_cifar10 import ModelHelper
from nets.resnet_at_ilsvrc12 import ModelHelper
#from nets.mobilenet_at_ilsvrc12 import ModelHelper

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_boolean('enbl_multi_gpu', False, 'enable multi-GPU training')
tf.app.flags.DEFINE_string('model_dir_in', './models', 'input model directory')
tf.app.flags.DEFINE_string('model_dir_out', './models_out', 'output model directory')
tf.app.flags.DEFINE_string('model_scope', 'model', 'model\'s variable scope name')
tf.app.flags.DEFINE_string('data_format', 'channels_last', 'data format in the output model')

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # create a TensorFlow session
    #sess = tf.Session()

    # create the model helper
    model_helper = ModelHelper(FLAGS.data_format)
    data_scope = 'data'
    model_scope = FLAGS.model_scope

    # bulid a graph with the target data format and rewrite checkpoint files
    with tf.Graph().as_default():
      # data input pipeline
      with tf.variable_scope(data_scope):
        iterator = model_helper.build_dataset_eval()
        images, __ = iterator.get_next()

      # model definition
      with tf.variable_scope(model_scope):
        logits = model_helper.forward_eval(images)

      # add input & output tensors to certain collections
      tf.add_to_collection('images_final', images)
      tf.add_to_collection('logits_final', logits)

      # restore variables from checkpoint files
      vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
      saver = tf.train.Saver(vars_all)
      save_path = tf.train.latest_checkpoint(FLAGS.model_dir_in)
      sess = tf.Session()
      saver.restore(sess, save_path)
      saver.save(sess, os.path.join(FLAGS.model_dir_out, 'model.ckpt'))

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
