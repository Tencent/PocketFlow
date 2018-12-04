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
"""Export a channel-pruned *.tflite model from checkpoint files."""

import os
import re
import traceback
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_string('model_dir', './models', 'model directory')
tf.app.flags.DEFINE_string('input_coll', 'images_final', 'input tensor\'s collection')
tf.app.flags.DEFINE_string('output_coll', 'logits_final', 'output tensor\'s collection')
tf.app.flags.DEFINE_boolean('enbl_fake_prune', False, 'enable fake pruning (for speed test only)')
tf.app.flags.DEFINE_float('fake_prune_ratio', 0.5, 'fake pruning ratio')
tf.app.flags.DEFINE_integer('nb_repts_warmup', 100, '# of repeated runs for warm-up')
tf.app.flags.DEFINE_integer('nb_repts', 100, '# of repeated runs for elapsed time measurement')

def get_file_path_meta():
  """Get the file path to the *.meta data.

  Returns:
  * file_path: file path to the *.meta data
  """

  pattern = re.compile('model.ckpt.meta$')
  for file_name in os.listdir(FLAGS.model_dir):
    if re.search(pattern, file_name) is not None:
      file_path = os.path.join(FLAGS.model_dir, file_name)
      break

  return file_path

def get_input_name_n_shape(file_path):
  """Get the input tensor's name & shape from *.meta file.

  Args:
  * file_path: file path to the *.meta data

  Returns:
  * input_name: input tensor's name
  * input_shape: input tensor's shape
  """

  with tf.Graph().as_default():
    tf.train.import_meta_graph(file_path)
    net_input = tf.get_collection(FLAGS.input_coll)[0]
    input_name = net_input.name
    input_shape = net_input.shape

  return input_name, input_shape

def get_data_format(sess):
  """Get the data format of convolutional layers.

  Args:
  * sess: TensorFlow session

  Returns:
  * data_format: data format of convolutional layers
  """

  data_format = None
  pattern = re.compile('Conv2D$')
  for op in tf.get_default_graph().get_operations():
    if re.search(pattern, op.name) is not None:
      data_format = op.get_attr('data_format').decode('utf-8')
      tf.logging.info('data format: ' + data_format)
      break

  return data_format

def convert_pb_model_to_tflite(file_path_pb, file_path_tflite, net_input_name, net_output_name):
  """Convert *.pb model to a *.tflite model.

  Args:
  * file_path_pb: file path to the *.pb model
  * file_path_tflite: file path to the *.tflite model
  * net_input_name: network's input node's name
  * net_output_name: network's output node's name
  """

  tf.logging.info(file_path_pb + ' -> ' + file_path_tflite)
  with tf.Graph().as_default():
    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
      file_path_pb, [net_input_name], [net_output_name])
    tflite_model = converter.convert()
    with tf.gfile.GFile(file_path_tflite, 'wb') as o_file:
      o_file.write(tflite_model)

def test_pb_model(file_path, net_input_name, net_output_name, net_input_data):
  """Test the *.pb model.

  Args:
  * file_path: file path to the *.pb model
  * net_input_name: network's input node's name
  * net_output_name: network's output node's name
  * net_input_data: network's input node's data
  """

  with tf.Graph().as_default() as graph:
    sess = tf.Session()

    # restore the model
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(file_path, 'rb') as i_file:
      graph_def.ParseFromString(i_file.read())
    tf.import_graph_def(graph_def)

    # obtain input & output nodes and then test the model
    net_input = graph.get_tensor_by_name('import/' + net_input_name + ':0')
    net_output = graph.get_tensor_by_name('import/' + net_output_name + ':0')
    tf.logging.info('input: {} / output: {}'.format(net_input.name, net_output.name))
    for idx in range(FLAGS.nb_repts_warmup + FLAGS.nb_repts):
      if idx == FLAGS.nb_repts_warmup:
        time_beg = timer()
      net_output_data = sess.run(net_output, feed_dict={net_input: net_input_data})
    time_elapsed = (timer() - time_beg) / FLAGS.nb_repts
    tf.logging.info('outputs from the *.pb model: {}'.format(net_output_data))
    tf.logging.info('time consumption of *.pb model: %.2f ms' % (time_elapsed * 1000))

def test_tflite_model(file_path, net_input_data):
  """Test the *.tflite model.

  Args:
  * file_path: file path to the *.tflite model
  * net_input_data: network's input node's data
  """

  # restore the model and allocate tensors
  interpreter = tf.contrib.lite.Interpreter(model_path=file_path)
  interpreter.allocate_tensors()

  # get input & output tensors
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  tf.logging.info('input details: {}'.format(input_details))
  tf.logging.info('output details: {}'.format(output_details))

  # test the model with given inputs
  for idx in range(FLAGS.nb_repts_warmup + FLAGS.nb_repts):
    if idx == FLAGS.nb_repts_warmup:
      time_beg = timer()
    interpreter.set_tensor(input_details[0]['index'], net_input_data)
    interpreter.invoke()
    net_output_data = interpreter.get_tensor(output_details[0]['index'])
  time_elapsed = (timer() - time_beg) / FLAGS.nb_repts
  tf.logging.info('outputs from the *.tflite model: {}'.format(net_output_data))
  tf.logging.info('time consumption of *.tflite model: %.2f ms' % (time_elapsed * 1000))

def is_initialized(sess, var):
  """Check whether a variable is initialized.

  Args:
  * sess: TensorFlow session
  * var: variabile to be checked
  """

  try:
    sess.run(var)
    return True
  except tf.errors.FailedPreconditionError:
    return False

def apply_fake_pruning(kernel):
  """Apply fake pruning to the convolutional kernel.

  Args:
  * kernel: original convolutional kernel

  Returns:
  * kernel: randomly pruned convolutional kernel
  """

  tf.logging.info('kernel shape: {}'.format(kernel.shape))
  nb_chns = kernel.shape[2]
  idxs_all = np.arange(nb_chns)
  np.random.shuffle(idxs_all)
  idxs_pruned = idxs_all[:int(nb_chns * FLAGS.fake_prune_ratio)]
  kernel[:, :, idxs_pruned, :] = 0.0

  return kernel

def replace_dropout_layers():
  """Replace dropout layers with identity mappings.

  Returns:
  * op_outputs_old: output nodes to be swapped in the old graph
  * op_outputs_new: output nodes to be swapped in the new graph
  """

  pattern_div = re.compile('/dropout/div')
  pattern_mul = re.compile('/dropout/mul')
  op_outputs_old, op_outputs_new = [], []
  for op in tf.get_default_graph().get_operations():
    if re.search(pattern_div, op.name) is not None:
      x = tf.identity(op.inputs[0])
      op_outputs_new += [x]
    if re.search(pattern_mul, op.name) is not None:
      op_outputs_old += [op.outputs[0]]

  return op_outputs_old, op_outputs_new

def insert_alt_routines(sess, graph_trans_mthd):
  """Insert alternative rountines for convolutional layers.

  Args:
  * sess: TensorFlow session
  * graph_trans_mthd: graph transformation method

  Returns:
  * op_outputs_old: output nodes to be swapped in the old graph
  * op_outputs_new: output nodes to be swapped in the new graph
  """

  pattern = re.compile('Conv2D$')
  op_outputs_old, op_outputs_new = [], []
  for op in tf.get_default_graph().get_operations():
    if re.search(pattern, op.name) is not None:
      # skip un-initialized variables, which is not needed in the final *.pb file
      if not is_initialized(sess, op.inputs[1]):
        continue

      # detect which channels to be pruned
      tf.logging.info('transforming OP: ' + op.name)
      kernel = sess.run(op.inputs[1])
      if FLAGS.enbl_fake_prune:
        kernel = apply_fake_pruning(kernel)
      kernel_chn_in = kernel.shape[2]
      strides = op.get_attr('strides')
      padding = op.get_attr('padding').decode('utf-8')
      data_format = op.get_attr('data_format').decode('utf-8')
      dilations = op.get_attr('dilations')
      nnzs = np.nonzero(np.sum(np.abs(kernel), axis=(0, 1, 3)))[0]
      tf.logging.info('reducing %d channels to %d' % (kernel_chn_in, nnzs.size))
      kernel_gthr = np.zeros((1, 1, kernel_chn_in, nnzs.size))
      kernel_gthr[0, 0, nnzs, np.arange(nnzs.size)] = 1.0
      kernel_shrk = kernel[:, :, nnzs, :]

      # replace channel pruned convolutional with cheaper operations
      if graph_trans_mthd == 'gather':
        x = tf.gather(op.inputs[0], nnzs, axis=1)
        x = tf.nn.conv2d(
          x, kernel_shrk, strides, padding, data_format=data_format, dilations=dilations)
      elif graph_trans_mthd == '1x1_conv':
        x = tf.nn.conv2d(op.inputs[0], kernel_gthr, [1, 1, 1, 1], 'SAME', data_format=data_format)
        x = tf.nn.conv2d(
          x, kernel_shrk, strides, padding, data_format=data_format, dilations=dilations)
      else:
        raise ValueError('unrecognized graph transformation method: ' + graph_trans_mthd)

      # obtain old and new routines' outputs
      op_outputs_old += [op.outputs[0]]
      op_outputs_new += [x]

  return op_outputs_old, op_outputs_new

def export_pb_tflite_model(net, file_path_meta, file_path_pb, file_path_tflite, edit_graph):
  """Export *.pb & *.tflite models from checkpoint files.

  Args:
  * net: network configurations
  * file_path_meta: file path to the *.meta data
  * file_path_pb: file path to the *.pb model
  * file_path_tflite: file path to the *.tflite model
  * edit_graph: whether the graph should be edited
  """

  # convert checkpoint files to a *.pb model
  with tf.Graph().as_default() as graph:
    sess = tf.Session()

    # restore the graph with inputs replaced
    net_input = tf.placeholder(tf.float32, shape=net['input_shape'], name=net['input_name'])
    saver = tf.train.import_meta_graph(
      file_path_meta, input_map={net['input_name_ckpt']: net_input})
    saver.restore(sess, file_path_meta.replace('.meta', ''))

    # obtain the data format and determine which graph transformation method to be used
    data_format = get_data_format(sess)
    graph_trans_mthd = 'gather' if data_format == 'NCHW' else '1x1_conv'

    # obtain the output node
    net_logits = tf.get_collection(FLAGS.output_coll)[0]
    net_output = tf.nn.softmax(net_logits, name=net['output_name'])
    tf.logging.info('input: {} / output: {}'.format(net_input.name, net_output.name))
    tf.logging.info('input\'s shape: {}'.format(net_input.shape))
    tf.logging.info('output\'s shape: {}'.format(net_output.shape))

    # replace dropout layers with identity mappings (TF-Lite does not support dropout layers)
    op_outputs_old, op_outputs_new = replace_dropout_layers()
    sess.close()
    graph_editor.swap_outputs(op_outputs_old, op_outputs_new)
    sess = tf.Session()  # open a new session
    saver.restore(sess, file_path_meta.replace('.meta', ''))

    # edit the graph by inserting alternative routines for each convolutional layer
    if edit_graph:
      op_outputs_old, op_outputs_new = insert_alt_routines(sess, graph_trans_mthd)
      sess.close()
      graph_editor.swap_outputs(op_outputs_old, op_outputs_new)
      sess = tf.Session()  # open a new session
      saver.restore(sess, file_path_meta.replace('.meta', ''))

    # write the original grpah to *.pb file
    graph_def = graph.as_graph_def()
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [net['output_name']])
    file_name_pb = os.path.basename(file_path_pb)
    tf.train.write_graph(graph_def, FLAGS.model_dir, file_name_pb, as_text=False)
    tf.logging.info(file_path_pb + ' generated')
    test_pb_model(file_path_pb, net['input_name'], net['output_name'], net['input_data'])

  # convert the *.pb model to a *.tflite model (only NHWC is supported)
  if data_format == 'NHWC':
    convert_pb_model_to_tflite(file_path_pb, file_path_tflite, net['input_name'], net['output_name'])
    tf.logging.info(file_path_tflite + ' generated')
    test_tflite_model(file_path_tflite, net['input_data'])
  else:
    tf.logging.warning('*.tflite model not generated since NCHW is not supported by TF-Lite')

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # network configurations
    file_path_meta = get_file_path_meta()
    input_name, input_shape = get_input_name_n_shape(file_path_meta)
    net = {
      'input_name_ckpt': input_name,  # used to import the model from checkpoint files
      'input_name': 'net_input',  # used to export the model to *.pb & *.tflite files
      'input_shape': input_shape,
      'output_name': 'net_output'
    }
    net['input_data'] = np.zeros(tuple([1] + list(net['input_shape'][1:])), dtype=np.float32)

    # generate *.pb & *.tflite files for the original model
    file_path_pb = os.path.join(FLAGS.model_dir, 'model_original.pb')
    file_path_tflite = os.path.join(FLAGS.model_dir, 'model_original.tflite')
    export_pb_tflite_model(net, file_path_meta, file_path_pb, file_path_tflite, edit_graph=False)

    # generate *.pb & *.tflite files for the transformed model
    file_path_pb = os.path.join(FLAGS.model_dir, 'model_transformed.pb')
    file_path_tflite = os.path.join(FLAGS.model_dir, 'model_transformed.tflite')
    export_pb_tflite_model(net, file_path_meta, file_path_pb, file_path_tflite, edit_graph=True)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
