#!/bin/bash

# obtain directory paths
if [ "$#" -eq 2 ]; then
  dir_curr="$1"
  dir_temp="$2"
else
  dir_curr=`pwd`
  dir_temp="${dir_curr}-minimal"
fi
echo "creating a minimal code directory: ${dir_curr} -> ${dir_temp}"

# copy all files to a temporary directory
rm -rf ${dir_temp}
cp -r ${dir_curr} ${dir_temp}
cd ${dir_temp}

# remove redundant files
#git clean -xdf  # all files ignored by git
rm -rf ./models*
rm -rf ./logs
rm -rf .git .gitignore
cp ${dir_curr}/path.conf .

# return to the original directory
cd ${dir_curr}
