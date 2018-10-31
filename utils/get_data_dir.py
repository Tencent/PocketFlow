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
"""Get data directory on the local machine."""

import re
import sys

# get the Python script's & path configuration file's path
assert len(sys.argv) == 3
py_file = sys.argv[1]
conf_file = sys.argv[2]

# obtain the dataset's name
pattern = re.compile(r'at_[0-9A-Za-z]+_run.py$')
match = re.search(pattern, py_file)
assert match is not None, 'unable to match pattern in ' + py_file
dataset_name = match.group(0).split('_')[1]

# extract local directory path to the dataset
pattern = re.compile(r'^([^#]*)#(.*)$')
with open(conf_file, 'r') as i_file:
  for i_line in i_file:
    # remove comments and whitespaces
    match = re.match(pattern, i_line)
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
    if key == 'data_dir_local_%s' % dataset_name:
      print(value)
      break
