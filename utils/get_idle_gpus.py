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
"""Get a list of idle GPUs."""

import os
import sys
import subprocess

# get the required number of idle GPUs
assert len(sys.argv) == 2
nb_idle_gpus = int(sys.argv[1])

# assume: idle gpu has no more than 50% of total card memory used
max_gpu_usage_threshold = .5

# command to execute to get gpu id and corresponding memory used
# and total memory. It gives output in the format
# gpu id, memory used, total memory
cmd = 'nvidia-smi --query-gpu=index,memory.used,memory.total ' \
  '--format=csv,noheader,nounits'
gpu_smi_output = subprocess.check_output(cmd, shell=True)
gpu_smi_output = gpu_smi_output.decode('utf-8')

idle_gpus = []

for gpu in gpu_smi_output.split(sep='\n')[:-1]:
  (gpu_id, used, total) = [int(value) for value in gpu.split(sep=',')]
  memory_usage_percentage = (used/total)
  if memory_usage_percentage < max_gpu_usage_threshold:
    # GPU usage is less than the threshold
    idle_gpus.append(gpu_id)

if len(idle_gpus) < nb_idle_gpus:
  raise ValueError('not enough idle GPUs;'
                   ' idle GPUs are: {}'.format(idle_gpus))
idle_gpus = idle_gpus[:nb_idle_gpus]
idle_gpus_str = ','.join([str(idle_gpu) for idle_gpu in idle_gpus])
print(idle_gpus_str)
