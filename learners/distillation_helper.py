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
"""Helper for training with distillation loss."""

import os
import shutil
import numpy as np
import tensorflow as tf

from utils.misc_utils import is_primary_worker

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('loss_w_dst', 4.0, 'distillation loss\'s multiplier')
tf.app.flags.DEFINE_float('tempr_dst', 4.0, 'temperature in the distillation loss')
tf.app.flags.DEFINE_string('save_path_dst', './models_dst/model.ckpt',
                           'distillation model\'s save path')

class DistillationHelper(object):
  """Helper for training with distillation loss.

  Other learners can use calc_loss() (remember to call initialize() first to make sure that the
    pre-trained model is available) to compute the distillation loss.
  """

  def __init__(self, sm_writer, model_helper, mpi_comm):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    * mpi_comm: MPI communication object
    """

    # initialize a full-precision model
    self.model_scope = 'distilled_model'  # to distinguish from models created by other learners
    from learners.full_precision.learner import FullPrecLearner
    self.learner = FullPrecLearner(sm_writer, model_helper, self.model_scope, enbl_dst=False)

    # initialize a model for training with the distillation loss
    if is_primary_worker('local'):
      self.__initialize()
    if FLAGS.enbl_multi_gpu:
      mpi_comm.Barrier()

  def calc_logits(self, sess, images):
    """Calculate the distilled model's logits for given images.

    Args:
    * sess: TensorFlow session to restore model weights
    * images: input images (shape: N x H x W x C)

    Returns:
    * logits: output logits (shape: N x K) of the distilled model

    Note: A new forward path will be built and initialized with pre-trained model's weights.
    """

    with tf.variable_scope(self.model_scope):
      # build a new forward path with given images
      logits = self.learner.forward_eval(images)
      logits = tf.stop_gradient(logits)  # prevent gradients to flow into the distilled model

      # initialize weights with the pre-trained model
      saver = tf.train.Saver(self.learner.vars)  # create a new saver for the new graph
      ckpt_file = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path_dst))
      saver.restore(sess, ckpt_file)
      tf.logging.info('model restored from ' + ckpt_file)

    return logits

  @classmethod
  def calc_loss(cls, logits_pri, logits_dst):
    """Calculate the distillation loss for the primary model's logits.

    Args:
      logits_pri: primary model's logits (shape: N x K)
      logits_dst: distilled model's logits (shape: N x K)

    Returns:
      loss: distillation loss
    """

    logits_soft = logits_pri / FLAGS.tempr_dst
    labels_soft = tf.nn.softmax(logits_dst / FLAGS.tempr_dst)
    loss = FLAGS.loss_w_dst * tf.losses.softmax_cross_entropy(labels_soft, logits_soft)
    tf.summary.scalar('distillation_loss', loss)

    return loss

  def __initialize(self):
    """Initialize a model for training with the distillation loss.

    Note: If the pre-trained model is not available on HDFS, then a new model will be trained from
        scratch and uploaded to HDFS.
    """

    # download the pre-trained model from HDFS
    self.learner.download_model()
    if os.path.isdir(os.path.dirname(FLAGS.save_path_dst)):
      shutil.rmtree(os.path.dirname(FLAGS.save_path_dst))
    shutil.copytree(os.path.dirname(FLAGS.save_path), os.path.dirname(FLAGS.save_path_dst))

    # restore a pre-trained model and then evaluate
    self.__restore()
    self.__evaluate()

  def __restore(self):
    """Restore a pre-trained model with the variable scope renamed."""

    # rename the variable scope
    ckpt_dir = os.path.dirname(FLAGS.save_path_dst)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    with tf.Graph().as_default():
      with tf.Session() as sess:
        # rename variables
        for var_name_old, __ in tf.contrib.framework.list_variables(ckpt_dir):
          var = tf.contrib.framework.load_variable(ckpt_dir, var_name_old)
          var_name_new = self.model_scope + '/' + '/'.join(var_name_old.split('/')[1:])
          var = tf.get_variable(var_name_new, initializer=var)
          tf.logging.info('renaming variable: {} -> {}'.format(var_name_old, var_name_new))

        # save renamed variables to the checkpoint file
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, ckpt.model_checkpoint_path)  # pylint: disable=no-member

    # restore the model from checkpoint files
    ckpt_file = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path_dst))
    self.learner.saver_eval.restore(self.learner.sess_eval, ckpt_file)
    tf.logging.info('model restored from ' + ckpt_file)

  def __evaluate(self):
    """Evaluate the model's loss & accuracy."""

    # evaluate the model
    losses, accuracies = [], []
    nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size_eval))
    for __ in range(nb_iters):
      eval_rslt = self.learner.sess_eval.run(self.learner.eval_op)
      losses.append(eval_rslt[0])
      accuracies.append(eval_rslt[1])
    tf.logging.info('loss: {}'.format(np.mean(np.array(losses))))
    tf.logging.info('accuracy: {}'.format(np.mean(np.array(accuracies))))
