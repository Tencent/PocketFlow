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
"""Utility functions."""

import os
import subprocess
import tensorflow as tf
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.contrib.lite.python import lite_constants as constants

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('uqtf_save_path_probe', './models_uqtf_probe/model.ckpt',
                           'UQ-TF: probe model\'s save path')

def insert_quant_op(graph, node_name, is_train):
  """Insert quantization operations to the specified activation node.

  Args:
  * graph: TensorFlow graph
  * node_name: activation node's name
  * is_train: insert training-related operations or not
  """

  # locate the node & activation operation
  for op in graph.get_operations():
    if node_name in [node.name for node in op.outputs]:
      tf.logging.info('op: {} / inputs: {} / outputs: {}'.format(
        op.name, [node.name for node in op.inputs], [node.name for node in op.outputs]))
      node = op.outputs[0]
      activation_op = op
      break

  # re-route the graph to insert quantization operations
  input_to_ops_map = input_to_ops.InputToOps(graph)
  consumer_ops = input_to_ops_map.ConsumerOperations(activation_op)
  node_quant = quant_ops.MovingAvgQuantize(
    node, is_training=is_train, num_bits=FLAGS.uqtf_activation_bits)
  nb_update_inputs = common.RerouteTensor(node_quant, node, consumer_ops)
  tf.logging.info('nb_update_inputs = %d' % nb_update_inputs)

def export_tflite_model(input_coll, output_coll, images_shape, images_name):
  """Export a *.tflite model from checkpoint files.

  Args:
  * input_coll: input collection's name
  * output_coll: output collection's name

  Returns:
  * unquant_node_name: unquantized activation node name (None if not found)
  """

  # remove previously generated *.pb & *.tflite models
  model_dir = os.path.dirname(FLAGS.uqtf_save_path_probe)
  pb_path = os.path.join(model_dir, 'model.pb')
  tflite_path = os.path.join(model_dir, 'model.tflite')
  if os.path.exists(pb_path):
    os.remove(pb_path)
  if os.path.exists(tflite_path):
    os.remove(tflite_path)

  # convert checkpoint files to a *.pb model
  images_name_ph = 'images'
  with tf.Graph().as_default() as graph:
    # create a TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # restore the graph with inputs replaced
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    meta_path = ckpt_path + '.meta'
    images = tf.placeholder(tf.float32, shape=images_shape, name=images_name_ph)
    saver = tf.train.import_meta_graph(meta_path, input_map={images_name: images})
    saver.restore(sess, ckpt_path)

    # obtain input & output nodes
    net_inputs = tf.get_collection(input_coll)
    net_outputs = tf.get_collection(output_coll)
    for node in net_inputs:
      tf.logging.info('inputs: {} / {}'.format(node.name, node.shape))
    for node in net_outputs:
      tf.logging.info('outputs: {} / {}'.format(node.name, node.shape))

    # write the original grpah to *.pb file
    graph_def = tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [node.name.replace(':0', '') for node in net_outputs])
    tf.train.write_graph(graph_def, model_dir, os.path.basename(pb_path), as_text=False)
    assert os.path.exists(pb_path), 'failed to generate a *.pb model'

  # convert the *.pb model to a *.tflite model
  tf.logging.info(pb_path + ' -> ' + tflite_path)
  arg_list = [
    '--graph_def_file ' + pb_path,
    '--output_file ' + tflite_path,
    '--input_arrays ' + images_name_ph,
    '--output_arrays ' + ','.join([node.name.replace(':0', '') for node in net_outputs]),
    '--inference_type QUANTIZED_UINT8',
    '--mean_values 128',
    '--std_dev_values 127']
  cmd_str = ' '.join(['tflite_convert'] + arg_list)
  with open('./dump', 'w') as o_file:
    subprocess.call(cmd_str, shell=True, stdout=o_file, stderr=o_file)

  # detect the unquantized activation node (if any)
  unquant_node_name = None
  if not os.path.exists(tflite_path):
    flag_str = 'tensorflow/contrib/lite/toco/tooling_util.cc:1634]'
    with open('./dump', 'r') as i_file:
      for i_line in i_file:
        if not 'is lacking min/max data' in i_line:
          continue
        for sub_line in i_line.split('\\n'):
          if flag_str in sub_line:
            sub_strs = sub_line.replace(',', ' ').split()
            unquant_node_name = sub_strs[sub_strs.index(flag_str) + 2] + ':0'
            break

  return unquant_node_name

def find_unquant_act_nodes(model_helper, data_scope, model_scope):
  """Find unquantized activation nodes in the model.

  TensorFlow's quantization-aware training APIs insert quantization operations into the graph,
    so that model weights can be fine-tuned with quantization error taken into consideration.
    However, these APIs only insert quantization operations into nodes matching certain topology
    rules, and some nodes may be left unquantized. When converting such model to *.tflite model,
    these unquantized nodes will introduce extra performance loss.
  Here, we provide a utility function to detect these unquantized nodes before training, so that
    quantization operations can be inserted. The resulting model can be smoothly exported to a
    *.tflite model.

  Args:
  * model_helper: model helper with definitions of model & dataset
  # data_scope: data scope name
  * model_scope: model scope name

  Returns:
  * unquant_node_names: list of unquantized activation node names
  """

  # obtain the image tensor's name & shape
  with tf.Graph().as_default():
    with tf.variable_scope(data_scope):
      iterator = model_helper.build_dataset_eval()
      inputs, labels = iterator.get_next()
      if not isinstance(inputs, dict):
        images_shape, images_name = inputs.shape, inputs.name
      else:
        images_shape, images_name = inputs['image'].shape, inputs['image'].name

  # create-quantize-save-export the model, and check for unquantized nodes
  input_coll = 'net_inputs'
  output_coll = 'net_outputs'
  unquant_node_names = []
  while True:
    # create a model, quantize it, and then save
    with tf.Graph().as_default() as graph:
      # create a TensorFlow session
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(data_scope):
        iterator = model_helper.build_dataset_eval()
        inputs, labels = iterator.get_next()

      # model definition - uniform quantized model
      with tf.variable_scope(model_scope):
        outputs = model_helper.forward_eval(inputs)
        tf.contrib.quantize.experimental_create_eval_graph(
          weight_bits=FLAGS.uqtf_weight_bits,
          activation_bits=FLAGS.uqtf_activation_bits,
          scope=model_scope)

        # manually insert quantization operations
        for node_name in unquant_node_names:
          insert_quant_op(graph, node_name, is_train=False)

      # add input & output tensors to collections
      if not isinstance(inputs, dict):
        tf.add_to_collection(input_coll, inputs)
      else:
        tf.add_to_collection(input_coll, inputs['image'])
      if not isinstance(outputs, dict):
        tf.add_to_collection(output_coll, outputs)
      else:
        for value in outputs.values():
          tf.add_to_collection(output_coll, value)

      # save the model
      vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_scope)
      saver = tf.train.Saver(vars_list)
      sess.run(tf.variables_initializer(vars_list))
      save_path = saver.save(sess, FLAGS.uqtf_save_path_probe)
      tf.logging.info('probe model saved to ' + save_path)

    # attempt to export a *.tflite model and detect unquantized activation nodes (if any)
    unquant_node_name = export_tflite_model(input_coll, output_coll, images_shape, images_name)
    if unquant_node_name:
      unquant_node_names += [unquant_node_name]
      tf.logging.info('node <%s> is not quantized' % unquant_node_name)
    else:
      break

  return unquant_node_names
