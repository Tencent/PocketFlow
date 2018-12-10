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
"""Measure the time consumption of *.pb and *.tflite models."""

import traceback
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_file', None, 'model file path')
tf.app.flags.DEFINE_string('input_name', 'net_input', 'input tensor\'s name in the *.pb model')
tf.app.flags.DEFINE_string('output_name', 'net_output', 'output tensor\'s name in the *.pb model')
tf.app.flags.DEFINE_string('input_dtype', 'float32',
                           'input tensor\'s data type in the *.tflite model')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size for run-time benchmark')
tf.app.flags.DEFINE_integer('nb_repts_warmup', 100, '# of repeated runs for warm-up')
tf.app.flags.DEFINE_integer('nb_repts', 100, '# of repeated runs for elapsed time measurement')

def test_pb_model():
  """Test the *.pb model."""

  with tf.Graph().as_default() as graph:
    sess = tf.Session()

    # restore the model
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.model_file, 'rb') as i_file:
      graph_def.ParseFromString(i_file.read())
    tf.import_graph_def(graph_def)

    # obtain input & output nodes and then test the model
    net_input = graph.get_tensor_by_name('import/' + FLAGS.input_name + ':0')
    net_output = graph.get_tensor_by_name('import/' + FLAGS.output_name + ':0')
    net_input_data = np.zeros(tuple([FLAGS.batch_size] + list(net_input.shape[1:])))
    for idx in range(FLAGS.nb_repts_warmup + FLAGS.nb_repts):
      if idx == FLAGS.nb_repts_warmup:
        time_beg = timer()
      sess.run(net_output, feed_dict={net_input: net_input_data})
    time_elapsed = (timer() - time_beg) / FLAGS.nb_repts / FLAGS.batch_size
    tf.logging.info('time consumption of *.pb model: %.2f ms' % (time_elapsed * 1000))

def test_tflite_model():
  """Test the *.tflite model."""

  # restore the model and allocate tensors
  interpreter = tf.contrib.lite.Interpreter(model_path=FLAGS.model_file)
  interpreter.allocate_tensors()

  # get input & output tensors
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  assert len(input_details) == 1, '<input_details> should contain only one element'
  if FLAGS.input_dtype == 'uint8':
    net_input_data = np.zeros(input_details[0]['shape'], dtype=np.uint8)
  elif FLAGS.input_dtype == 'float32':
    net_input_data = np.zeros(input_details[0]['shape'], dtype=np.float32)
  else:
    raise ValueError('unrecognized input data type: ' + FLAGS.input_dtype)

  # test the model with given inputs
  for idx in range(FLAGS.nb_repts_warmup + FLAGS.nb_repts):
    if idx == FLAGS.nb_repts_warmup:
      time_beg = timer()
    interpreter.set_tensor(input_details[0]['index'], net_input_data)
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])
  time_elapsed = (timer() - time_beg) / FLAGS.nb_repts
  tf.logging.info('time consumption of *.tflite model: %.2f ms' % (time_elapsed * 1000))

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # call benchmark routines for *.pb / *.tflite models
    if FLAGS.model_file is None:
      raise ValueError('<FLAGS.model_file> must not be None')
    elif FLAGS.model_file.endswith('.pb'):
      test_pb_model()
    elif FLAGS.model_file.endswith('.tflite'):
      test_tflite_model()
    else:
      raise ValueError('unrecognized model file path: ' + FLAGS.model_file)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
