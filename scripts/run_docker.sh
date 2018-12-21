#!/bin/bash

# obtain directory paths
dir_curr=`pwd`
dir_temp="${dir_curr}-minimal"

# create a minimal code directory
./scripts/create_minimal.sh ${dir_curr} ${dir_temp}
cd ${dir_temp}

# default arguments
nb_gpus=1

# parse arguments passed from the command line
py_script="$1"
shift
extra_args=""
for i in "$@"
do
  case "$i" in
    -n=*|--nb_gpus=*)
    nb_gpus="${i#*=}"
    shift
    ;;
    *)
    # unknown option
    extra_args="${extra_args} ${i}"
    shift
    ;;
  esac
done
extra_args_path=`python utils/get_path_args.py docker ${py_script} path.conf`
extra_args="${extra_args} ${extra_args_path}"
echo ${extra_args} > extra_args
echo "Python script: ${py_script}"
echo "Data directory: ${dir_data}"
echo "# of GPUs: ${nb_gpus}"
echo "extra arguments: ${extra_args}"

# create temporary directory for log & model
dir_temp_log=`mktemp -d`
dir_temp_model=`mktemp -d`
dir_temp_data=`python utils/get_data_dir.py ${py_script} path.conf`

# enter the docker environment
cp -v ${py_script} main.py
idle_gpus=`python utils/get_idle_gpus.py ${nb_gpus}`
NV_GPU="${idle_gpus}" nvidia-docker run -it --rm --hostname=local-env \
    --env SEVEN_HTTP_FORWARD_PORT= --env NB_GPUS=${nb_gpus} --network=host \
    -v ${dir_temp}:/opt/ml/env \
    -v ${dir_temp_log}:/opt/ml/log \
    -v ${dir_temp_model}:/opt/ml/model \
    -v ${dir_temp_data}:/opt/ml/data \
    -w /opt/ml/env \
    docker.oa.com/g_tfplus/horovod:python3.5 bash

# return to the main directory
rm -rf ${dir_temp_log} ${dir_temp_model}
cd ${dir_curr}
