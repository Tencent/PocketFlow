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
"""Convert AutoML-generated hyper-parameter file to PocketFlow-compatible format."""

import sys

# file paths
assert len(sys.argv) == 2, '[HELP] python cvt_hparam_file.py <file_path>'
file_path = sys.argv[1]

# read hyper-parameters' values from file
with open(file_path, 'r') as i_file:
  # obtain raw hyper-parameters' values
  for i_line in i_file:
    sub_strs = i_line.split()
    name, val = sub_strs[0], float(sub_strs[2])
    if name == 'ws_prune_ratio_exp':
      ws_prune_ratio_exp = val
    elif name == 'ws_iter_ratio_beg':
      ws_iter_ratio_beg = val
    elif name == 'ws_iter_ratio_end':
      ws_iter_ratio_end = val
    elif name == 'ws_update_mask_step':
      ws_update_mask_step = val

  # make sure <iter_ratio_beg> is smaller than <iter_ratio_end>
  ws_iter_ratio_end = ws_iter_ratio_beg + ws_iter_ratio_end * (1.0 - ws_iter_ratio_beg)

# write hyper-parameters' values to file
output_str = ''
output_str += ' --ws_prune_ratio_exp %.4f' % ws_prune_ratio_exp
output_str += ' --ws_iter_ratio_beg %.4f' % ws_iter_ratio_beg
output_str += ' --ws_iter_ratio_end %.4f' % ws_iter_ratio_end
output_str += ' --ws_update_mask_step %d' % int(ws_update_mask_step)
print(output_str)
