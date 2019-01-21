#!/bin/bash

# configure pip to use internal source
unset http_proxy
unset https_proxy
mkdir -p ~/.pip/ \
    && echo "[global]"                                              > ~/.pip/pip.conf \
    && echo "index-url = http://mirror-sng.oa.com/pypi/web/simple/" >> ~/.pip/pip.conf \
    && echo "trusted-host = mirror-sng.oa.com"                      >> ~/.pip/pip.conf
cat ~/.pip/pip.conf

# install Python packages with Internet access
pip install tensorflow-gpu==1.12.0
pip install horovod
pip install docopt
pip install hdfs
pip install scipy
pip install sklearn
pip install pandas
pip install mpi4py

# add the current directory to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:`pwd`
export LD_LIBRARY_PATH=/opt/ml/disk/local/cuda/lib64:$LD_LIBRARY_PATH

# start TensorBoard
LOG_DIR=/opt/ml/log
mkdir -p ${LOG_DIR}
nohup tensorboard \
    --port=${SEVEN_HTTP_FORWARD_PORT} \
    --host=127.0.0.1 \
    --logdir=${LOG_DIR} \
    >/dev/null 2>&1 &

# execute the main script
mkdir models
EXTRA_ARGS=`cat ./extra_args`
if [ ${NB_GPUS} -eq 1 ]; then
  echo "multi-GPU training disabled"
  python main.py --log_dir ${LOG_DIR} ${EXTRA_ARGS}
elif [ ${NB_GPUS} -le 8 ]; then
  echo "multi-GPU training enabled"
  options="-np ${NB_GPUS} -H localhost:${NB_GPUS} -bind-to none -map-by slot
      -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1
      -x LD_LIBRARY_PATH --mca btl_tcp_if_include eth1"
  mpirun ${options} python main.py --enbl_multi_gpu --log_dir ${LOG_DIR} ${EXTRA_ARGS}
fi

# archive model files to HDFS
mv models* /opt/ml/model

# remove *.pyc files
find . -name "*.pyc" -exec rm -f {} \;
