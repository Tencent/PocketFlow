# Installation

PocketFlow is developed and tested on Linux, using Python 3.6 and TensorFlow 1.10.0. We support the following three execution modes for PocketFlow:

* Local mode: run PocketFlow on the local machine.
* Docker mode: run PocketFlow within a docker image.
* Seven mode: run PocketFlow on the seven cluster (only available within Tencent).

## Clone PocketFlow

To make a local copy of the PocketFlow repository, use:

``` bash
$ git clone https://github.com/Tencent/PocketFlow.git
```

## Create a Path Configuration File

PocketFlow requires a path configuration file, named `path.conf`, to setup directory paths to data sets and pre-trained models under different execution modes, as well as HDFS / HTTP connection parameters.

We have provided a template file to help you create your own path configuration file. You can find it in the PocketFlow repository, named `path.conf.template`, which contains more detailed descriptions on how to customize path configurations. For instance, if you want to use CIFAR-10 and ImageNet data sets stored on the local machine, then the path configuration file should look like this:

``` bash
# data files
data_hdfs_host = None
data_dir_local_cifar10 = /home/user_name/datasets/cifar-10-batches-bin  # this line has been edited!
data_dir_hdfs_cifar10 = None
data_dir_seven_cifar10 = None
data_dir_docker_cifar10 = /opt/ml/data  # DO NOT EDIT
data_dir_local_ilsvrc12 = /home/user_name/datasets/imagenet_tfrecord  # this line has been edited!
data_dir_hdfs_ilsvrc12 = None
data_dir_seven_ilsvrc12 = None
data_dir_docker_ilsvrc12 = /opt/ml/data  # DO NOT EDIT

# model files
model_http_url = https://api.ai.tencent.com/pocketflow
```

In short, you need to replace "None" in the template file with the actual path (or HDFS / HTTP connection parameters) if available, or leave it unchanged otherwise.

## Prepare for the Local Mode

We recommend to use Anaconda as the Python environment, which has many essential packages built-in. The Anaconda installer can be downloaded from [here](https://www.anaconda.com/download/#linux). To install, use the following command:

``` bash
# install Anaconda; replace the installer's file name if needed
$ bash Anaconda3-5.2.0-Linux-x86_64.sh

# activate Anaconda's Python path
$ source ~/.bashrc
```

For Anaconda 5.3.0 or later, the default Python version is 3.7, which does not support installing TensorFlow with pip directly. Therefore, you need to manually switch to Python 3.6 once Anaconda is installed:

``` bash
# install Python 3.6
$ conda install python=3.6
```

To install TensorFlow, you may refer to TensorFlow's official [documentation](https://www.tensorflow.org/install/pip) for detailed instructions. Specially, if GPU-based training is required, then you need to follow the [GPU support guide](https://www.tensorflow.org/install/gpu) to set up a CUDA-enabled GPU card in prior to installation. After that, install TensorFlow with:

``` bash
# TensorFlow with GPU support; use <tensorflow> if GPU is not available
$ pip install tensorflow-gpu

# verify the install
$ python -c "import tensorflow as tf; print(tf.__version__)"
```

To run PocketFlow in the local mode, *e.g.* to train a full-precision ResNet-20 model for the CIFAR-10 classification task, use the following command:

``` bash
$ ./scripts/run_local.sh nets/resnet_at_cifar10_run.py
```

## Prepare for the Docker Mode

Docker offers an alternative way to run PocketFlow within an isolated container, so that your local Python environment remains untouched. We recommend you to use the [horovod](https://github.com/uber/horovod) docker provided by Uber, which enables multi-GPU distributed training for TensorFlow with only a few lines modification. Once docker is installed, the docker image can be obtained via:

``` bash
# obtain the docker image
$ docker pull uber/horovod
```

To run PocketFlow in the docker mode, *e.g.* to train a full-precision ResNet-20 model for the CIFAR-10 classification task, use the following command:

``` bash
$ ./scripts/run_docker.sh nets/resnet_at_cifar10_run.py
```

## Prepare for the Seven Mode

Seven is a distributed learning platform built for both CPU and GPU clusters. Users can submit tasks to the seven cluster, using built-in data sets and docker images seamlessly.

To run PocketFlow in the seven mode, *e.g.* to train a full-precision ResNet-20 model for the CIFAR-10 classification task, use the following command:

``` bash
$ ./scripts/run_seven.sh nets/resnet_at_cifar10_run.py
```
