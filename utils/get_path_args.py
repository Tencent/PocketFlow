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
"""Get path-related arguments."""

import re
import sys

# parse input arguments
assert len(sys.argv) == 4
exec_mode = sys.argv[1]
py_file = sys.argv[2]
conf_file = sys.argv[3]

# obtain the dataset's name
pattern = re.compile(r'at_[0-9A-Za-z]+_run.py$')
match = re.search(pattern, py_file)
assert match is not None, 'unable to match pattern in ' + py_file
dataset_name = match.group(0).split('_')[1]

# extract path-related arguments from path.conf
arg_list = []
pattern_comment = re.compile(r'^([^#]*)#(.*)$')
pattern_data_dir = re.compile(r'^data_dir_[a-z]+_%s$' % dataset_name)
data_dirs = {'local': None, 'docker': None, 'seven': None, 'hdfs': None}
with open(conf_file, 'r') as i_file:
  for i_line in i_file:
    # remove comments and whitespaces
    match = re.match(pattern_comment, i_line)
    if match:
      i_line = match.group(1)
    i_line = i_line.strip()  # remote whitespaces
    if i_line == '':
      continue

    # extract the (key, value) pair
    key_n_value = i_line.split(' = ')
    assert len(key_n_value) == 2, 'each line must contains exactly one \' = \''
    key = key_n_value[0].strip()
    value = key_n_value[1].strip()
    if value == 'None':
      continue

    # extract arguments
    if not key.startswith('data_dir_'):
      arg_list += ['--%s %s' % (key, value)]
    elif re.match(pattern_data_dir, key):
      data_disk = key.split('_')[2]
      data_dirs[data_disk] = value

# append path-related arguments
if exec_mode in ['local', 'seven'] and data_dirs[exec_mode] is not None:
  arg_list += ['--data_dir_local %s' % data_dirs[exec_mode]]
elif data_dirs['local'] is not None:
  arg_list += ['--data_dir_local %s' % data_dirs['docker']]
if data_dirs['hdfs'] is not None:
  arg_list += ['--data_dir_hdfs %s' % data_dirs['hdfs']]

# concatenate all arguments into one string
arg_list_str = ' '.join(arg_list)
print(arg_list_str)
