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
"""Execution script for ResNet models on the ILSVRC-12 dataset."""

import traceback
import tensorflow as tf

from nets.resnet_at_ilsvrc12 import ModelHelper
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
