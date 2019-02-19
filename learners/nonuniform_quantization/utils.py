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
""" Util fnctions for Non-Uniform Quantization """

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


def prefix_filter(prefix):
  """ filter out the variable_scope """
  ind = prefix.index('/')
  return prefix[ind+1:]


class NonUniformQuantization:
  # pylint: disable=too-many-instance-attributes
  """ Class of non-uniform quantization """

  def __init__(self,
               sess,
               bucket_size=0,
               use_buckets=False,
               init_style='quantile',
               bucket_type='split'):
    self.sess = sess
    self.bucket_size = bucket_size
    self.use_buckets = use_buckets
    self.bucket_type = bucket_type
    self.init_style = init_style
    self.matmul_ops = []
    self.activation_ops = []
    self.quantized_matmul_ops = []
    self.quantized_activation_ops = []
    self.init_style = init_style
    self.quant_fn = self.__nonuni_quantize if not self.use_buckets else self.__bucket_quantize
    self.bucket_storage = tf.constant(0, dtype=tf.int32)  # bits
    self.__safe_check()

    # TODO: add more types of activations and mul
    self.support_act_types = ['Relu', 'Relu6', 'Crelu', 'Elu', 'Selu', 'Softplus',\
        'Softsign', 'Sigmoid', 'Tanh']
    self.support_mul_types = ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']

  def insert_quant_op_for_activations(self, act_bit_dict):
    """ Insert quantization operation for activation

    Args:
    * act_bit_dict: A dict with (key: act_op_name, value: act_bits)
    """
    activation_fn = {'Relu': tf.nn.relu,
                     'Tanh': tf.nn.tanh,
                     'Softplus': tf.nn.softplus,
                     'Sigmoid': tf.nn.sigmoid,
                     'Relu6': tf.nn.relu6}

    for op in self.activation_ops:
      old_sgv = ge.sgv(op)
      input_ = old_sgv.inputs[0]
      if op.type in self.support_act_types:
        try:
          tmp_input_ = activation_fn[op.type](input_)
        except KeyError:
          raise NotImplementedError("The activation_fn needs to include %s manually" % op.type)

        prefix = prefix_filter(op.name)
        qa = self.__uniform_quantize(tmp_input_, act_bit_dict[op.name], 'activation', prefix)
        new_sgv = ge.sgv(qa.op)
        ge.reroute_outputs(new_sgv, old_sgv)
        self.quantized_activation_ops.append(qa.op)
      else:
        raise NotImplementedError("Unknown activation mode, you may add it manually here")

  def insert_quant_op_for_weights(self, w_bit_dict):
    """Insert quantization operation for weights

    Args:
    * wewight_bit_dict: A dict with (key: matmul_op_name, value: quant_bits)
    """
    for op in self.matmul_ops:
      w = op.inputs[1]
      prefix = prefix_filter(op.name)
      qw = self.quant_fn(w, w_bit_dict[op.name], 'weight', prefix)
      weight_fn = {'MatMul': tf.matmul,
                   'Conv2D': tf.nn.conv2d,
                   'DepthwiseConv2dNative': tf.nn.depthwise_conv2d}
      is_conv_fn = lambda x: 'Conv' in x.type
      try:
        if is_conv_fn(op):
          strides = op.get_attr('strides')
          padding = op.get_attr('padding')
          qw_op = weight_fn[op.type](op.inputs[0], qw, strides, padding).op
        else:
          # fc layers
          qw_op = weight_fn[op.type](op.inputs[0], qw).op
      except KeyError:
        raise NotImplementedError("Unrecognied Mul op, \
            try to add it into matmul_typs for quantization")

      self.quantized_matmul_ops.append(qw_op)

    # replace input
    for wop, qwop in zip(self.matmul_ops, self.quantized_matmul_ops):
      old_sgv = ge.sgv(wop)
      new_sgv = ge.sgv(qwop)
      ge.reroute_inputs(new_sgv, old_sgv)

  def search_matmul_op(self, quantize_all_layers):
    """ search matmul or Conv2D operations in graph for quantization"""

    is_student_fn = lambda x: 'distilled' not in x.name
    for op in self.sess.graph.get_operations():
      if op.type in self.support_mul_types and is_student_fn(op):
        self.matmul_ops.append(op)
    if not quantize_all_layers:
      self.matmul_ops = self.matmul_ops[1:-1]

    return self.matmul_ops

  def search_activation_op(self):
    """ search activation operation in graph for quantization"""

    is_student_fn = lambda x: 'distilled' not in x.name
    for op in self.sess.graph.get_operations():
      if op.type in self.support_act_types and is_student_fn(op):
        self.activation_ops.append(op)
    return self.activation_ops

  def get_layerwise_tune_op(self, var, lrn_rate=1e-3):
    """ Get the layerwise fine-tuning ops

    Returns:
    * A list of ops for fine-tuning with len(matmul_ops) elements
    """
    layerwise_diff = []
    tune_ops = []
    for (v, q_op) in zip(var, self.quantized_matmul_ops):
      inputs = q_op.inputs[0]
      quant_outputs = q_op.outputs[0]
      # TODO: wrap it into a function, as also used above.
      if 'MatMul' in q_op.type:
        fp_outputs = tf.matmul(inputs, v)
      elif 'Conv2D' in q_op.type:
        strides = q_op.get_attr('strides')
        padding = q_op.get_attr('padding')
        fp_outputs = tf.nn.conv2d(inputs, v, strides, padding)
      else:
        raise ValueError("Unrecognized Mul Op")

      diff = tf.reduce_mean(tf.square(quant_outputs - fp_outputs))
      tune_ops.append(tf.train.AdamOptimizer(lrn_rate).minimize(diff, var_list=v))
      layerwise_diff.append(diff)
    return tune_ops, layerwise_diff

  def __nonuni_quantize(self, x, mbits, mode, prefix=''):
    """ Non-uniform quantization function

    Args:
    * x: A Tensor to be quantized.
    * mbits: A Tensor, specifying the quantization bits.
    * mode: 'weight' or 'activation', indicating where to quantize
    * prefix: A 'string' indicating the variable scope.

    Returns:
    * A 'Tensor' with the same shape of 'x' that is non-uniformly quantized.
    """

    with tf.variable_scope(prefix + '/nonuniform_quantize'):
      k = tf.cast(2 ** mbits, dtype=tf.int64)
      x_normalized, alpha, beta = self.__scale(x, mode)
      if self.init_style == 'quantile':
        init_c = self.__quantile_init(x_normalized, k)
      elif self.init_style == 'uniform':
        init_c = self.__uniform_init(k)
      else:
        raise ValueError("Unrecognized Initialization Mode.")
      qx = self.__build_norm_quant_point(init_c, x_normalized, k)

      inv_qx = self.__inv_scale(qx, alpha, beta)
      print("Quantized: " + tf.get_variable_scope().name)
      return inv_qx

  def __bucket_quantize(self, x, mbits, mode, prefix=''):
    """ Non-uniform quantization function with bucketing

    Args:
    * x: A 'Tensor'.
    * mbits: A Tensor, indicating the quantization bits.
    * mode: A string, 'weight' or 'activation', indicating where to quantize
    * bucket_size: A 'scalar' indicating the size for each bucket
    * prefix: A 'string' indicating the variable scope.

    Returns:
    * A 'Tensor' with the same shape of 'x' that is non-uniformly quantized.
    """
    with tf.variable_scope(prefix + '/nonuniform_bucket_quantize'):
      k = tf.cast(2 ** mbits, tf.int64)
      orig_shape = x.get_shape()
      if self.bucket_type == 'split' and mode == 'weight':
        x, bucket_num, padded_num = self.__split_bucket(x)  # [bucket_size, bucket_num]
      elif self.bucket_type == 'channel' and mode == 'weight':
         # [bucket_size, bucket_num], padded_num=0
        x, bucket_num, padded_num = self.__channel_bucket(x)
      else:
        # do not use bucket, orig shape
        pass
      x_normalized, alpha, beta = self.__scale(x, mode) # [bucket_size, bucket_num]

      if self.init_style == 'quantile' and mode == 'weight':
        init_c = self.__quantile_init(x_normalized, k)  # [nb_clusters, bucket_number]
      elif self.init_style == 'uniform' and mode == 'weight':
        init_c = self.__uniform_init(x_normalized, k) # [nb_clusters, bucket_number]
      else:
        raise ValueError('Unrecognized Initialization Mode.')

      qx = self.__build_bucket_norm_quant_point(init_c, x_normalized, k, bucket_num)

      inv_qx = self.__inv_scale(qx, alpha, beta)
      inv_qx = tf.reshape(inv_qx, [-1])
      if padded_num != 0:
        inv_qx = tf.reshape(inv_qx[:-padded_num], orig_shape)  # recover back the quantized weights.
      else:
        inv_qx = tf.reshape(inv_qx, orig_shape)

      print("Quantized: " + tf.get_variable_scope().name + prefix)
      # Update bucket storage if use buckets.
      if mode == 'weight':
        self.__updt_bucket_storage(bucket_num)

    return inv_qx

  def __uniform_quantize(self, x, mbits, mode, prefix=''):
    """Uniform quantization function

    Args:
    * x: A Tensor (weights or activation output)
    * mbits: A scalar Tensor, tf.int64, spicifying number of bit for quantization
    * mode: A string, 'weight' or 'activation', where to quantize
    * prefix: A string, the prefix of scope name

    Returns:
    * A Tensor, uniform quantized value
    """
    with tf.variable_scope(prefix + '/uniform_quantize'):

      if self.use_buckets and mode == 'weight':
        orig_shape = x.get_shape()
        if self.bucket_type == 'split':
          x, bucket_num, padded_num = self.__split_bucket(x)
        elif self.bucket_type == 'channel':
          x, bucket_num, padded_num = self.__channel_bucket(x)
      x_normalized, alpha, beta = self.__scale(x, mode)
      g = self.sess.graph
      k = tf.cast(2 ** mbits - 1, tf.float32)
      with g.gradient_override_map({'Round': 'Identity'}):
        qw = tf.round(x_normalized * k) / k
      qw = self.__inv_scale(qw, alpha, beta)
      if self.use_buckets and mode == 'weight':
        # Reshape w back to the original shape
        qw = tf.reshape(qw, [-1])
        if padded_num != 0:
          qw = tf.reshape(qw[:-padded_num], orig_shape)
        else:
          qw = tf.reshape(qw, orig_shape)

       # Update bucket storage if use buckets.
        self.__updt_bucket_storage(bucket_num)
      print("Quantized: " + tf.get_variable_scope().name)
      return qw

  def __build_norm_quant_point(self, init_c, x_normalized, k):
    """ Build the quantization points on [0, 1] and quantize 'x_normalized',
        the function applies for no_bucket and channel-wise bucket.

    Args:
    * init_c: A Tensor, the initialization of quantization points
    * x_normalized: A Tensor, the normalized weights
    * k: A constant Tensor, the number of quantization points

    Returns:
    * A 'Tensor' with the same shape of 'x_normalized',
        denoting the quantized value for x_normalized
    """
    c = tf.get_variable('clusters', validate_shape=False, initializer=init_c)
    w_dims = x_normalized.shape.__len__()
    shape_ = tf.concat((tf.ones(w_dims, dtype=tf.int64), [k]), axis=0)
    g = self.sess.graph
    w_new = tf.tile(tf.expand_dims(x_normalized, w_dims), shape_)
    min_index = tf.argmin(tf.abs(w_new - c), axis=-1)

    # override gradient for the STE estimator
    with g.gradient_override_map({'Mul': 'Add', 'Sign': 'Identity'}):
      qx = tf.gather(c, min_index) * tf.sign(x_normalized + 1e-6)
    return qx

  def __build_bucket_norm_quant_point(self, init_c, x_normalized, k, bucket_num):
    """ Build the quantization points on [0, 1] and quantize 'x_normalized',
        the function applies for both type 'split' and 'channel'.

    Args:
    * init_c: A Tensor, the initialization of quantization points, [nb_cluster, bucket_num]
    * x_normalized: A Tensor, the normalized weights, [bucket_size, bucket_num]
    * k: A constant Tensor, the number of quantization points
    * bucket_type: A Tensor, specifying the number of buckets

    Returns:
    * A 'Tensor' with the same shape of 'x_normalized',
        denoting the quantized value for x_normalized
    """

    c = tf.get_variable('clusters', validate_shape=False, initializer=init_c)
    w_dims = x_normalized.shape.__len__()
    shape_ = tf.concat((tf.ones(w_dims, dtype=tf.int64), [k]), axis=0)
    g = self.sess.graph
    x_rep = tf.tile(tf.expand_dims(x_normalized, -1), shape_)
    x_rep = tf.transpose(x_rep, [0, 2, 1])  # [bucket_size, nb_cluster, bucket_num]

    # Non uniform: assign each w to the closest cluster
    min_index = tf.argmin(tf.abs(x_rep - c), axis=1)  # [bucket_size, bucket_num]

    # NOTE: slow but save memory
    tmp_qx = tf.map_fn(lambda idx: tf.gather(c[:, idx], min_index[:, idx]), \
        tf.range(bucket_num), dtype=tf.float32)
    # NOTE: anotehr way, quick but consume memory
    #tmp_qx = tf.gather(c, min_index)
    #tmp_qx = tf.transpose(tmp_qx, [1,2,0])
    #tmp_qx = tf.gather_nd(tmp_qx, list(zip(range(bucket_num), range(bucket_num))))

    qx = tf.transpose(tmp_qx) # [bucket_size, bucket_num]

    # override gradient for the STE estimator
    with g.gradient_override_map({'Mul': 'Add', 'Sign': 'Identity'}):
      qx = qx * tf.sign(x_normalized + 1e-6)
    return qx

  def __quantile_init(self, x_normalized, nb_clusters):
    """ Use quantiles of weights to initialize the quantization points.
        If bucketing is enabled, both 'split' and 'channel' applies.

    Args:
    * x_normalized: A 'Tensor' that are normalized. The shape is 1-D if not use_buckets
        else 2-D with '[bucket_size, 'bucket_num]'.
    * nb_clusters: A Tensor, indicating the number of quantization points.

    Returns:
    * A 'Tensor'. The shape is 1-D with 'nb_clusters' elements if not use_buckets
        else 2-D with '[nb_clusters, bucket_number]'.
    """
    axis = None if not self.use_buckets else 0
    init_c = tf.map_fn(lambda idx: tf.contrib.distributions.percentile(\
        x_normalized, (idx+1) * 100 / (nb_clusters+1), axis=axis), \
        tf.range(nb_clusters), dtype=tf.float32)
    return init_c

  def __uniform_init(self, nb_clusters, bucket_num=0):
    """ Uniformly initialize the quantization points.
        If bucketing is enabled, both 'split' and 'channel' applies.

    Args:
    * nb_clusters: A Tensor, indicating the number of quantization points.
    * bucket_num: A Tensor, the number of buckets. Calculated differently for 'split' and 'channel'.

    Returns:
    * A 'Tensor'. The shape is 1-D with 'nb_clusters' elements if not use_buckets
        else 2-D with '[nb_clusters, bucket_number]'.
    """

    # uniformly initialized
    init_c = tf.linspace(0., 1., nb_clusters)
    if self.use_buckets:
      # [nb_clusters, bucket_num]
      init_c = tf.transpose(tf.reshape(tf.tile(init_c, [bucket_num]), [-1, nb_clusters]))
    return init_c

  def __scale(self, w, mode):
    """linear scale function

    Args:
    * w: A Tensor (weights or activation output),
         the shape is [bucket_size, bucekt_num] if use_bucket else the original size.
    * mode: A string, 'weight' or 'activation'

    Returns:
    * A Tensor, the normalized weights
    * A Tensor, alpha, scalar if activation mode else a vector [bucket_num].
    * A Tensor, beta, scalar if activation mode else a vector [bucket_num].
    """
    if mode == 'weight':
      if self.use_buckets:
        axis = 0
      else:
        axis = None
    elif mode == 'activation':
      axis = None
    else:
      raise ValueError("Unknown mode for scalling")

    w_max = tf.stop_gradient(tf.reduce_max(w, axis=axis))
    w_min = tf.stop_gradient(tf.reduce_min(w, axis=axis))
    eps = tf.constant(value=1e-10, dtype=tf.float32)

    alpha = w_max - w_min + eps
    beta = w_min
    w = (w - beta) / alpha
    return w, alpha, beta

  def __inv_scale(self, w, alpha, beta):
    """Inversed linear scale function

    Args:
    * w: A Tensor (weights or activations), origin shape
         if not use_buckets else [bucket_size, bucket_num]
    * alpha: A float value, scale factor if in activation mode else a vector [bucket_num]
    * bete: A float value, scale bias if in activation mode else a vector [bucket_num]

    Returns:
    * A Tensor, inversed scale value
    """

    return alpha * w + beta

  def __split_bucket(self, w):
    """ reshape weights according to bucket for 'split' type.
    Args:
    * w: A Tensor (weights)

    Returns:
    * A Tensor, shape [bucket_size, bucket_num]
    * An integer: the number of buckets
    * An integer: the number of padded elements
    """

    flat_w = tf.reshape(w, [-1])
    num_w = flat_w.get_shape()[0].value
    # use the last value to fill
    fill_value = flat_w[-1]

    multiple, rest = divmod(num_w, self.bucket_size)
    if rest != 0:
      values_to_add = tf.ones(self.bucket_size - rest) * fill_value
      # add the fill_value to make the tensor into a multiple of the bucket size.
      flat_w = tf.concat([flat_w, values_to_add], axis=0)
      multiple += 1

    flat_w = tf.reshape(flat_w, [self.bucket_size, -1])
    padded_num = (self.bucket_size - rest) if rest != 0 else 0

    return flat_w, multiple, padded_num

  def __channel_bucket(self, w):
    """ reshape weights according to bucket for 'channel' type.
        Note that for fc layers, buckets are created row-wisely.
    Args:
      w: A Tensor (weights)

    Returns:
      A Tensor shape [bucket_size, bucket_num], bucket_size = h*w*cin for conv or cin for fc
      A integer: the number of buckets
      A integer (0), zero padded elements
    """
    cout = w.get_shape()[-1].value
    folded_w = tf.reshape(w, [-1, cout])
    return folded_w, cout, 0

  def __safe_check(self):
    """ TODO: Check the name of bucket_type, the value of bucket_size """
    if self.bucket_size < 0:
      raise ValueError("Bucket size must be a postive integer")
    if self.bucket_type != 'split' and self.bucket_type != 'channel':
      raise ValueError("Unrecognized bucket type, must be 'weight' or 'channel'.")
    if self.init_style != 'uniform' and self.init_style != 'quantile':
      raise ValueError("Unrecognized initialization style")

  def __updt_bucket_storage(self, bucket_num):
    """ Calculate extra storage for the bucket scalling factors

    Args:
    * alpha: a Tensor, the scalling factor
    * beta: a Tensor, the scalling factor
    """
    self.bucket_storage += bucket_num * 32 * 2 # both alpha and beta, so *2.

