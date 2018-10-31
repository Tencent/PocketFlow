#!/bin/bash

# obtain directory paths
dir_curr=`pwd`
dir_temp="${dir_curr}-minimal"

# create a minimal code directory
./scripts/create_minimal.sh ${dir_curr} ${dir_temp}

# create a *.tar.gz archive and then download it
dir_name_temp=`basename ${dir_temp}`
tgz_name="${dir_name_temp}.tar.gz"
cd ..
tar -czvf ${tgz_name} ${dir_name_temp}
sz -b ${tgz_name}
cd ${dir_curr}
