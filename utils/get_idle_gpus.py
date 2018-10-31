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

# dump the output of "nvidia-smi" command to file
dump_file = './nvidia-smi-dump'
with open(dump_file, 'w') as o_file:
  subprocess.call(['nvidia-smi'], stdout=o_file)

# parse the output of "nvidia-smi" command
with open(dump_file, 'r') as i_file:
  # obtain list of all & busy GPUs
  parse_procs = False
  all_gpus, busy_gpus = [], []
  for i_line in i_file:
    if 'Processes' in i_line:
      parse_procs = True
    sub_strs = i_line.split()
    if len(sub_strs) < 2:
      continue
    if not parse_procs:
      if sub_strs[1].isdigit():
        all_gpus.append(sub_strs[1])
    else:
      if sub_strs[1].isdigit():
        busy_gpus.append(sub_strs[1])

  # obtain list of idle GPUs
  idle_gpus = list(set(all_gpus) - set(busy_gpus))
  idle_gpus.sort()
  if len(idle_gpus) < nb_idle_gpus:
    raise ValueError('not enough idle GPUs; idle GPUs are: {}'.format(idle_gpus))
  idle_gpus = idle_gpus[:nb_idle_gpus]
  idle_gpus_str = ','.join([str(idle_gpu) for idle_gpu in idle_gpus])
  print(idle_gpus_str)

# remove the dump file
os.remove(dump_file)
