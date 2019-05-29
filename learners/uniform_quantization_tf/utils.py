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
from tensorflow.lite.python import lite_constants

from utils.misc_utils import auto_barrier
from utils.misc_utils import is_primary_worker
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('uqtf_save_path_probe', './models_uqtf_probe/model.ckpt',
                           'UQ-TF: probe model\'s save path')
tf.app.flags.DEFINE_string('uqtf_save_path_probe_eval', './models_uqtf_probe_eval/model.ckpt',
                           'UQ-TF: probe model\'s save path for evaluation')

def create_session():
  """Create a TensorFlow session.

  Return:
  * sess: TensorFlow session
  """

  # create a TensorFlow session
  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
  config.gpu_options.allow_growth = True  # pylint: disable=no-member
  sess = tf.Session(config=config)

  return sess

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
  model_dir = os.path.dirname(FLAGS.uqtf_save_path_probe_eval)
  idx_worker = mgw.local_rank() if FLAGS.enbl_multi_gpu else 0
  pb_path = os.path.join(model_dir, 'model_%d.pb' % idx_worker)
  tflite_path = os.path.join(model_dir, 'model_%d.tflite' % idx_worker)
  if os.path.exists(pb_path):
    os.remove(pb_path)
  if os.path.exists(tflite_path):
    os.remove(tflite_path)

  # convert checkpoint files to a *.pb model
  images_name_ph = 'images'
  with tf.Graph().as_default() as graph:
    # create a TensorFlow session
    sess = create_session()

    # restore the graph with inputs replaced
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    meta_path = ckpt_path + '.meta'
    images = tf.placeholder(tf.float32, shape=images_shape, name=images_name_ph)
    saver = tf.train.import_meta_graph(meta_path, input_map={images_name: images})
    saver.restore(sess, ckpt_path)

    # obtain input & output nodes
    net_inputs = tf.get_collection(input_coll)
    net_logits = tf.get_collection(output_coll)[0]
    net_outputs = [tf.nn.softmax(net_logits)]
    for node in net_inputs:
      tf.logging.info('inputs: {} / {}'.format(node.name, node.shape))
    for node in net_outputs:
      tf.logging.info('outputs: {} / {}'.format(node.name, node.shape))

    # write the original grpah to *.pb file
    graph_def = tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [node.name.replace(':0', '') for node in net_outputs])
    tf.train.write_graph(graph_def, model_dir, os.path.basename(pb_path), as_text=False)
    assert os.path.exists(pb_path), 'failed to generate a *.pb model'

  # convert the *.pb model to a *.tflite model and detect the unquantized activation node (if any)
  tf.logging.info(pb_path + ' -> ' + tflite_path)
  converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    pb_path, [images_name_ph], [node.name.replace(':0', '') for node in net_outputs])
  converter.inference_type = lite_constants.QUANTIZED_UINT8
  converter.quantized_input_stats = {images_name_ph: (0., 1.)}
  unquant_node_name = None
  try:
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as o_file:
      o_file.write(tflite_model)
  except Exception as err:
    err_msg = str(err)
    flag_str = 'tensorflow/contrib/lite/toco/tooling_util.cc:1634]'
    for sub_line in err_msg.split('\\n'):
      if flag_str in sub_line:
        sub_strs = sub_line.replace(',', ' ').split()
        unquant_node_name = sub_strs[sub_strs.index(flag_str) + 2] + ':0'
        break
    assert unquant_node_name is not None, 'unable to locate the unquantized node'

  return unquant_node_name

def build_graph(model_helper, unquant_node_names, config, is_train):
  """Build a graph for training or evaluation.

  Args:
  * model_helper: model helper with definitions of model & dataset
  * unquant_node_names: list of unquantized activation node names
  * config: graph configuration
  * is_train: insert training-related operations or not

  Returns:
  * model: dictionary of model-related objects & operations
  """

  # setup function handles
  if is_train:
    build_dataset_fn = model_helper.build_dataset_train
    forward_fn = model_helper.forward_train
    create_quant_graph_fn = tf.contrib.quantize.experimental_create_training_graph
  else:
    build_dataset_fn = model_helper.build_dataset_eval
    forward_fn = model_helper.forward_eval
    create_quant_graph_fn = tf.contrib.quantize.experimental_create_eval_graph

  # build a graph for trianing or evaluation
  model = {}
  with tf.Graph().as_default() as graph:
    # data input pipeline
    with tf.variable_scope(config['data_scope']):
      iterator = build_dataset_fn()
      inputs, __ = iterator.get_next()

    # model definition - uniform quantized model
    with tf.variable_scope(config['model_scope']):
      # obtain outputs from model's forward-pass
      outputs = forward_fn(inputs)
      if not isinstance(outputs, dict):
        outputs_sfmax = tf.nn.softmax(outputs)  # <outputs> is logits
      else:
        outputs_sfmax = tf.nn.softmax(outputs['cls_pred'])  # <outputs['cls_pred']> is logits

      # quantize the graph using TensorFlow APIs
      create_quant_graph_fn(
        weight_bits=FLAGS.uqtf_weight_bits,
        activation_bits=FLAGS.uqtf_activation_bits,
        scope=config['model_scope'])

      # manually insert quantization operations
      for node_name in unquant_node_names:
        insert_quant_op(graph, node_name, is_train=is_train)

      # randomly increase each trainable variable's value
      incr_ops = []
      for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        incr_ops += [var.assign_add(tf.random.uniform(var.shape))]
      incr_op = tf.group(incr_ops)

    # add input & output tensors to collections
    if not isinstance(inputs, dict):
      tf.add_to_collection(config['input_coll'], inputs)
    else:
      tf.add_to_collection(config['input_coll'], inputs['image'])
    if not isinstance(outputs, dict):
      tf.add_to_collection(config['output_coll'], outputs)
    else:
      tf.add_to_collection(config['output_coll'], outputs['cls_pred'])

    # save the model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=config['model_scope'])
    model['sess'] = create_session()
    model['saver'] = tf.train.Saver(vars_list)
    model['init_op'] = tf.variables_initializer(vars_list)
    model['incr_op'] = incr_op

  return model

def find_unquant_act_nodes(model_helper, data_scope, model_scope, mpi_comm):
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
  * data_scope: data scope name
  * model_scope: model scope name
  * mpi_comm: MPI communication object

  Returns:
  * unquant_node_names: list of unquantized activation node names
  """

  # setup configurations
  config = {
    'data_scope': data_scope,
    'model_scope': model_scope,
    'input_coll': 'inputs',
    'output_coll': 'outputs',
  }

  # obtain the image tensor's name & shape
  with tf.Graph().as_default():
    with tf.variable_scope(data_scope):
      iterator = model_helper.build_dataset_eval()
      inputs, labels = iterator.get_next()
      if not isinstance(inputs, dict):
        images_shape, images_name = inputs.shape, inputs.name
      else:
        images_shape, images_name = inputs['image'].shape, inputs['image'].name

  # iteratively check for unquantized nodes
  unquant_node_names = []
  while True:
    # build training & evaluation graphs
    model_train = build_graph(model_helper, unquant_node_names, config, is_train=True)
    model_eval = build_graph(model_helper, unquant_node_names, config, is_train=False)

    # initialize a model in the training graph, and then save
    model_train['sess'].run(model_train['init_op'])
    model_train['sess'].run(model_train['incr_op'])
    save_path = model_train['saver'].save(model_train['sess'], FLAGS.uqtf_save_path_probe)
    tf.logging.info('model saved to ' + save_path)

    # restore a model in the evaluation graph from *.ckpt files, and then save again
    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.uqtf_save_path_probe))
    model_eval['saver'].restore(model_eval['sess'], save_path)
    tf.logging.info('model restored from ' + save_path)
    save_path = model_eval['saver'].save(model_eval['sess'], FLAGS.uqtf_save_path_probe_eval)
    tf.logging.info('model saved to ' + save_path)

    # try to export *.tflite models and check for unquantized nodes (if any)
    unquant_node_name = export_tflite_model(
      config['input_coll'], config['output_coll'], images_shape, images_name)
    if unquant_node_name:
      unquant_node_names += [unquant_node_name]
      tf.logging.info('node <%s> is not quantized' % unquant_node_name)
    else:
      break

  return unquant_node_names
