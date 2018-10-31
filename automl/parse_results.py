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
"""Convert PocketFlow-generated result file to AutoML-compatible format."""

import sys

assert len(sys.argv) == 2, '[HELP] python parse_results.py <file_path>'
file_path = sys.argv[1]

with open(file_path, 'r') as i_file:
  for i_line in i_file:
    if 'INFO:tensorflow:accuracy:' in i_line:
      accuracy = float(i_line.split()[-1])
    elif 'INFO:tensorflow:pruning ratio:' in i_line:
      prune_ratio = float(i_line.split()[-1])
    elif 'INFO:tensorflow:loss:' in i_line:
      loss = float(i_line.split()[-1])

print('object_value=%f' % accuracy)
print('prune_ratio=%f' % prune_ratio)
print('loss=%f' % loss)
