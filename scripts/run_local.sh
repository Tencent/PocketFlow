#!/bin/bash

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
extra_args_path=`python utils/get_path_args.py local ${py_script} path.conf`
extra_args="${extra_args} ${extra_args_path}"
echo "Python script: ${py_script}"
echo "# of GPUs: ${nb_gpus}"
echo "extra arguments: ${extra_args}"

# obtain list of idle GPUs
idle_gpus=`python utils/get_idle_gpus.py ${nb_gpus}`
export CUDA_VISIBLE_DEVICES=${idle_gpus}

# re-create the logging directory
rm -rf logs && mkdir logs

# execute the specified Python script with one or more GPUs
cp -v ${py_script} main.py
if [ ${nb_gpus} -eq 1 ]; then
  echo "multi-GPU training disabled"
  python main.py ${extra_args}
elif [ ${nb_gpus} -le 8 ]; then
  echo "multi-GPU training enabled"
  options="-np ${nb_gpus} -H localhost:${nb_gpus} -bind-to none -map-by slot
      -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1
      -x LD_LIBRARY_PATH --mca btl_tcp_if_include eth1"
  mpirun ${options} python main.py --enbl_multi_gpu ${extra_args}
fi
