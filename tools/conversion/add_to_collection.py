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
"""Add a list of tensors to specified collections (useful when exporting *.pb & *.tflite models)."""

import os
import re
import traceback
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir_in', './models_in', 'input model directory')
tf.app.flags.DEFINE_string('model_dir_out', './models_out', 'output model directory')
tf.app.flags.DEFINE_string('tensor_names', None, 'list of tensors names (comma-separated)')
tf.app.flags.DEFINE_string('coll_names', None, 'list of collection names (comma-separated)')

'''
Example: SSD (VGG-16) @ Pascal VOC

Input:
* data/IteratorGetNext:1 / (?, 300, 300, 3) / images
Output:
* quant_model/ssd300/multibox_head/cls_5/Conv2D:0 / (?, 1, 1, 84) / cls_preds
* quant_model/ssd300/multibox_head/loc_5/Conv2D:0 / (?, 1, 1, 16) / loc_preds
'''

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # add a list of tensors to specified collections
    with tf.Graph().as_default() as graph:
      # create a TensorFlow session
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)

      # restore a model from *.ckpt files
      ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir_in)
      meta_path = ckpt_path + '.meta'
      saver = tf.train.import_meta_graph(meta_path)
      saver.restore(sess, ckpt_path)

      # parse tensor & collection names
      tensor_names = [sub_str.strip() for sub_str in FLAGS.tensor_names.split(',')]
      coll_names = [sub_str.strip() for sub_str in FLAGS.coll_names.split(',')]
      assert len(tensor_names) == len(coll_names), \
        '# of tensors and collections does not match: %d (tensor) vs. %d (collection)' \
        % (len(tensor_names), len(coll_names))

      # obtain the full list of tensors in the graph
      tensors = set()
      for op in graph.get_operations():
        tensors |= set(op.inputs) | set(op.outputs)
      tensors = list(tensors)
      tensors.sort(key=lambda x: x.name)

      # find tensors and add them to corresponding collections
      for tensor in tensors:
        if tensor.name in tensor_names:
          tf.logging.info('tensor: {} / {}'.format(tensor.name, tensor.shape))
          coll_name = coll_names[tensor_names.index(tensor.name)]
          tf.add_to_collection(coll_name, tensor)
          tf.logging.info('added tensor <{}> to collection <{}>'.format(tensor.name, coll_name))

      # save the modified model
      vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      saver_new = tf.train.Saver(vars_list)
      save_path = saver_new.save(sess, os.path.join(FLAGS.model_dir_out, 'model.ckpt'))
      tf.logging.info('model saved to ' + save_path)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
