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
"""Export *.pb & *.tflite models from checkpoint files.

Description:
* To export compressed *.pb & *.tflite models trained with channel pruning based algorithms,
    set <enbl_chn_prune> to True.
* To export compressed *.pb & *.tflite models trained with the <UniformQuantTFLearner> learner,
    set <enbl_uni_quant> to True.
"""

import os
import re
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor
from tensorflow.contrib.lite.python import lite_constants

FLAGS = tf.app.flags.FLAGS

# common configurations
tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_string('model_dir', './models', 'model directory')
tf.app.flags.DEFINE_string('input_coll', 'images_final', 'input tensor\'s collection')
tf.app.flags.DEFINE_string('output_coll', 'logits_final', 'output tensor\'s collection')

# channel-pruning-related configurations
tf.app.flags.DEFINE_boolean('enbl_chn_prune', False,
                            'enable exporting models with pruned channels removed')
tf.app.flags.DEFINE_boolean('enbl_fake_prune', False, 'enable fake pruning (for speed test only)')
tf.app.flags.DEFINE_float('fake_prune_ratio', 0.5, 'fake pruning ratio')

# uniform-quantization-related configurations
tf.app.flags.DEFINE_boolean('enbl_uni_quant', False,
                            'enable exporting models with uniform quantization operations applied')
tf.app.flags.DEFINE_boolean('enbl_fake_quant', False,
                            'enable post-training quantization (may have extra performance loss)')

def get_meta_path():
  """Get the path to the *.meta file.

  Returns:
  * file_path: path to the *.meta file
  """

  pattern = re.compile(r'model\.ckpt\.meta$')  # file name must be: *model.ckpt.meta
  for file_name in os.listdir(FLAGS.model_dir):
    if re.search(pattern, file_name) is not None:
      file_path = os.path.join(FLAGS.model_dir, file_name)
      break

  return file_path

def get_input_name_n_shape(meta_path):
  """Get the input tensor's name & shape from *.meta file.

  Args:
  * meta_path: path to the *.meta file

  Returns:
  * input_name: input tensor's name
  * input_shape: input tensor's shape
  """

  with tf.Graph().as_default():
    tf.train.import_meta_graph(meta_path)
    net_input = tf.get_collection(FLAGS.input_coll)[0]
    input_name = net_input.name
    input_shape = net_input.shape

  return input_name, input_shape

def get_data_format():
  """Get the data format of convolutional layers.

  Returns:
  * data_format: data format of convolutional layers
  """

  data_format = None
  pattern = re.compile(r'Conv2D$')
  for op in tf.get_default_graph().get_operations():
    if re.search(pattern, op.name) is not None:
      data_format = op.get_attr('data_format').decode('utf-8')
      tf.logging.info('data format: ' + data_format)
      break
  assert data_format is not None, 'unable to determine <data_format>; convolutional layer not found'

  return data_format

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

def convert_pb_model_to_tflite(net, pb_path, tflite_path):
  """Convert the *.pb model to a *.tflite model.

  Args:
  * net: network configurations
  * pb_path: path to the *.pb file
  * tflite_path: path to the *.tflite file
  """

  # setup a TFLite converter
  tf.logging.info(pb_path + ' -> ' + tflite_path)
  converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    pb_path, [net['input_name']], [net['output_name']])
  if FLAGS.enbl_uni_quant:
    converter.inference_type = lite_constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {net['input_name']: (0., 1.)}
  if FLAGS.enbl_fake_quant:
    converter.post_training_quantize = True
    converter.default_ranges_stats = (0, 6)

  # convert the *.pb model to a *.tflite model
  try:
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as o_file:
      o_file.write(tflite_model)
    tf.logging.info(tflite_path + ' generate')
  except Exception as err:
    tf.logging.info('unable to generate a *.tflite model')
    raise err

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
    net_output_data = sess.run(net_output, feed_dict={net_input: net_input_data})
    tf.logging.info('outputs from the *.pb model: {}'.format(net_output_data))

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
  if not FLAGS.enbl_uni_quant:
    interpreter.set_tensor(input_details[0]['index'], net_input_data)
  else:
    interpreter.set_tensor(input_details[0]['index'], net_input_data.astype(np.uint8))
  interpreter.invoke()
  net_output_data = interpreter.get_tensor(output_details[0]['index'])
  tf.logging.info('outputs from the *.tflite model: {}'.format(net_output_data))

def export_pb_tflite_model(net, meta_path, pb_path, tflite_path):
  """Export *.pb & *.tflite models from checkpoint files.

  Args:
  * net: network configurations
  * meta_path: path to the *.meta file
  * pb_path: path to the *.pb file
  * tflite_path: path to the *.tflite file
  """

  # convert checkpoint files to a *.pb model
  with tf.Graph().as_default() as graph:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=no-member
    sess = tf.Session(config=config)

    # restore the graph with inputs replaced
    net_input = tf.placeholder(tf.float32, shape=net['input_shape'], name=net['input_name'])
    saver = tf.train.import_meta_graph(
      meta_path, input_map={net['input_name_ckpt']: net_input})
    saver.restore(sess, meta_path.replace('.meta', ''))

    # obtain the data format and determine which graph transformation method to be used
    data_format = get_data_format()
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
    sess = tf.Session(config=config)  # open a new session
    saver.restore(sess, meta_path.replace('.meta', ''))

    # edit the graph by inserting alternative routines for each convolutional layer
    if FLAGS.enbl_chn_prune:
      op_outputs_old, op_outputs_new = insert_alt_routines(sess, graph_trans_mthd)
      sess.close()
      graph_editor.swap_outputs(op_outputs_old, op_outputs_new)
      sess = tf.Session(config=config)  # open a new session
      saver.restore(sess, meta_path.replace('.meta', ''))

    # write the original grpah to *.pb file
    graph_def = graph.as_graph_def()
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [net['output_name']])
    file_name_pb = os.path.basename(pb_path)
    tf.train.write_graph(graph_def, FLAGS.model_dir, file_name_pb, as_text=False)
    tf.logging.info(pb_path + ' generated')

  # convert the *.pb model to a *.tflite model
  convert_pb_model_to_tflite(net, pb_path, tflite_path)

  # test *.pb & *.tflite models
  test_pb_model(pb_path, net['input_name'], net['output_name'], net['input_data'])
  test_tflite_model(tflite_path, net['input_data'])

def main(unused_argv):
  """Main entry.

  Args:
  * unused_argv: unused arguments (after FLAGS is parsed)
  """

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # network configurations
    meta_path = get_meta_path()
    input_name, input_shape = get_input_name_n_shape(meta_path)
    net = {
      'input_name_ckpt': input_name,  # used to import the model from checkpoint files
      'input_name': 'net_input',  # used to export the model to *.pb & *.tflite files
      'input_shape': input_shape,
      'output_name': 'net_output'
    }
    net['input_data'] = np.random.random(size=tuple([1] + list(net['input_shape'])[1:]))

    # generate *.pb & *.tflite files
    pb_path = os.path.join(FLAGS.model_dir, 'model.pb')
    tflite_path = os.path.join(FLAGS.model_dir, 'model.tflite')
    export_pb_tflite_model(net, meta_path, pb_path, tflite_path)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
