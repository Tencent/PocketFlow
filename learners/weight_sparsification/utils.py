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
"""Utility functions for the weight sparsification learner."""

def get_maskable_vars(trainable_vars):
  """Get a list of maskable variables.

  Args:
  * trainable_vars: list of trainable variables

  Returns:
  * maskable_vars: list of maskable variables

  Kernels in the following layer types will be matched:
  * tf.layers.conv2d
  * tf.layers.dense
  * Pointwise convolutional layer in slim.separable_conv2d
  """

  vars_kernel = [var for var in trainable_vars if 'kernel' in var.name]
  vars_ptconv = [var for var in trainable_vars if 'pointwise/weights' in var.name]
  vars_fnconv = [var for var in trainable_vars if 'Conv2d_1c_1x1/weights' in var.name]
  maskable_vars = vars_kernel + vars_ptconv + vars_fnconv

  return maskable_vars
