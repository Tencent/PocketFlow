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
"""Wrapper for multi-GPU training."""

# use Hovorod / TF-Plus for multi-GPU training
try:
  import horovod.tensorflow as mgw
  print('using Horovod for multi-GPU training')
except ImportError:
  try:
    import tfplus.tensorflow as mgw
    print('using TF-Plus for multi-GPU training')
  except ImportError:
    print('[WARNING] TF-Plus & Horovod cannot be imported; multi-GPU training is unsupported')

class MultiGpuWrapper(object):
  """Wrapper for multi-GPU training."""

  def __init__(self):
    """Constructor function."""
    pass

  @classmethod
  def init(cls, *args):
    """Initialization."""

    try:
      return mgw.init(*args)
    except NameError:
      raise NameError('module <mgw> not imported')

  @classmethod
  def size(cls, *args):
    """Get the number of workers at all nodes."""

    try:
      return mgw.size(*args)
    except NameError:
      raise NameError('module <mgw> not imported')

  @classmethod
  def rank(cls, *args):
    """Get the rank of current worker at all nodes."""

    try:
      return mgw.rank(*args)
    except NameError:
      raise NameError('module <mgw> not imported')

  @classmethod
  def local_size(cls, *args):
    """Get the number of workers at the current node."""

    try:
      return mgw.local_size(*args)
    except NameError:
      raise NameError('module <mgw> not imported')

  @classmethod
  def local_rank(cls, *args):
    """Get the rank of current worker at the current node."""

    try:
      return mgw.local_rank(*args)
    except NameError:
      raise NameError('module <mgw> not imported')

  @classmethod
  def DistributedOptimizer(cls, *args):
    """Get a distributed optimizer from the base optimizer."""

    try:
      return mgw.DistributedOptimizer(*args)
    except NameError:
      raise NameError('module <mgw> not imported')

  @classmethod
  def broadcast_global_variables(cls, *args):
    """Get a TensorFlow operation to broadcast all the global variables."""

    try:
      return mgw.broadcast_global_variables(*args)
    except NameError:
      raise NameError('module <mgw> not imported')
