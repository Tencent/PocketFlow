# Tutorial

In this tutorial, we demonstrate how to compress a convolutional neural network and export the compressed model into a \*.tflite file for deployment on mobile devices. The model we used here is a 18-layer residual network (denoted as "ResNet-18") trained for the ImageNet classification task. We will compress it with the discrimination-aware channel pruning algorithm (Zhuang et al., NIPS '18) to reduce the number of convolutional channels used in the network for speed-up.

## Prepare the Data

To start with, we need to convert the ImageNet data set (ILSVRC-12) into TensorFlow's native TFRecord file format. You may follow the data preparation guide [here](https://github.com/tensorflow/models/tree/master/research/inception#getting-started) to download the full data set and convert it into TFRecord files. After that, you should be able to find 1,024 training files and 128 validation files in the data directory, like this:

``` bash
# training files
train-00000-of-01024
train-00001-of-01024
...
train-01023-of-01024

# validation files
validation-00000-of-00128
validation-00001-of-00128
...
validation-00127-of-00128
```

## Prepare the Pre-trained Model

The discrimination-aware channel pruning algorithm requires a pre-trained uncompressed model provided in advance, so that a channel-pruned model can be trained with warm-start. You can download a pre-trained model from [here](https://api.ai.tencent.com/pocketflow/list.html), and then unzip files into the `models` sub-directory.

Alternatively, you can train an uncompressed full-precision model from scratch using `FullPrecLearner` with the following command (choose whatever mode that fits you):

``` bash
# local mode with 1 GPU
$ ./scripts/run_local.sh nets/resnet_at_ilsvrc12_run.py

# docker mode with 8 GPUs
$ ./scripts/run_docker.sh nets/resnet_at_ilsvrc12_run.py -n=8

# seven mode with 8 GPUs
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py -n=8
```

After the training process, you should be able to find the resulting model files located at the `models` sub-directory in PocketFlow's home directory.

## Train the Compressed Model

Now, we can train a compressed model with the discrimination-aware channel pruning algorithm, as implemented by `DisChnPrunedLearner`. Assuming you are now in PocketFlow's home directory, the training process of model compression can be started using the following command (choose whatever mode that fits you):

``` bash
# local mode with 1 GPU
$ ./scripts/run_local.sh nets/resnet_at_ilsvrc12_run.py \
    --learner dis-chn-pruned

# docker mode with 8 GPUs
$ ./scripts/run_docker.sh nets/resnet_at_ilsvrc12_run.py -n=8 \
    --learner dis-chn-pruned

# seven mode with 8 GPUs
$ ./scripts/run_seven.sh nets/resnet_at_ilsvrc12_run.py -n=8 \
    --learner dis-chn-pruned
```

Let's take the execution command for the local mode as an example. In this command, `run_local.sh` is a shell script that executes the specified Python script with user-provided arguments. Here, we ask it to run the Python script named `nets/resnet_at_ilsvrc12_run.py`, which is the execution script for ResNet models on the ImageNet data set. After that, we use `--learner dis-chn-pruned` to specify that the `DisChnPrunedLearner` should be used for model compression. You may also use other learners by specifying the corresponding learner name. Below is a full list of available learners in PocketFlow:

| Learner name     | Learner class            | Note                                                                          |
|:-----------------|:-------------------------|:------------------------------------------------------------------------------|
| `full-prec`      | `FullPrecLearner`        | No model compression                                                          |
| `channel`        | `ChannelPrunedLearner`   | Channel pruning with LASSO-based channel selection (He et al., 2017)          |
| `dis-chn-pruned` | `DisChnPrunedLearner`    | Discrimination-aware channel pruning (Zhuang et al., 2018)                    |
| `weight-sparse`  | `WeightSparseLearner`    | Weight sparsification with dynamic pruning schedule (Zhu & Gupta, 2017)       |
| `uniform`        | `UniformQuantLearner`    | Weight quantization with uniform reconstruction levels (Jacob et al., 2018)   |
| `uniform-tf`     | `UniformQuantTFLearner`  | Weight quantization with uniform reconstruction levels and TensorFlow APIs    |
| `non-uniform`    | `NonUniformQuantLearner` | Weight quantization with non-uniform reconstruction levels (Han et al., 2016) |

The local mode only uses 1 GPU for the training process, which takes approximately 20-30 hours to complete. This can be accelerated by multi-GPU training in the docker and seven mode, which is enabled by adding `-n=x` right after the specified Python script, where `x` is the number of GPUs to be used.

Optionally, you can pass some extra arguments to customize the training process. For the discrimination-aware channel pruning algorithm, some of key arguments are:

| Name              | Definition                             | Default Value |
|:------------------|:---------------------------------------|:--------------|
| `enbl_dst`        | Enable training with distillation loss | False         |
| `dcp_prune_ratio` | DCP algorithm's pruning ratio          | 0.5           |

You may override the default value by appending customized arguments at the end of the execution command. For instance, the following command:

``` bash
$ ./scripts/run_local.sh nets/resnet_at_ilsvrc12_run.py \
    --learner dis-chn-pruned \
    --enbl_dst \
    --dcp_prune_ratio 0.75
```

requires the `DisChnPrunedLearner` to achieve an overall pruning ratio of 0.75 and the training process will be carried out with the distillation loss. As a result, the number of channels in each convolutional layer of the compressed model will be one quarter of the original one.

After the training process is completed, you should be able to find a sub-directory named `models_dcp_eval` created in the home directory of PocketFlow. This sub-directory contains all the files that define the compressed model, and we will export them to a TensorFlow Lite formatted model file for deployment in the next section.

## Export to TensorFlow Lite

TensorFlow's checkpoint files cannot be directly used for deployment on mobile devices. Instead, we need to firstly convert them into a single \*.tflite file that is supported by the TensorFlow Lite Interpreter. For model compressed with channel-pruning based algorithms, *e.g.* `ChannelPruningLearner` and `DisChnPrunedLearner`, we have prepared a model conversion script, `tools/conversion/export_pb_tflite_models.py`, to generate a TF-Lite model from TensorFlow's checkpoint files.

To convert checkpoint files into a \*.tflite file, use the following command:

``` bash
# convert checkpoint files into a *.tflite model
$ python tools/conversion/export_pb_tflite_models.py \
    --model_dir models_dcp_eval
```

In the above command, we specify the model directory containing checkpoint files generated in the previous training process. The conversion script automatically detects which channels can be safely pruned, and then produces a light-weighted compressed model. The resulting TensorFlow Lite file is also placed at the `models_dcp_eval` directory, named as `model_transformed.tflite`.

## Deploy on Mobile Devices

After exporting the compressed model to the TensorFlow Lite file format, you may follow the official [guide](https://www.tensorflow.org/lite/demo_android) for creating an Android demo App from it. Basically, this demo App uses a TensorFlow Lite model to continuously classifies images captured by the camera, and all the computation are performed on mobile devices in real time.

To use the `model_transformed.tflite` model file, you need to place it in the `asserts` directory and create a Java class named `ImageClassifierFloatResNet` to use this model for classification. Below is the example code, which is modified from `ImageClassifierFloatInception.java` used in the official demo project:

``` Java
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo;

import android.app.Activity;

import java.io.IOException;

/**
 * This classifier works with the ResNet-18 model.
 * It applies floating point inference rather than using a quantized model.
 */
public class ImageClassifierFloatResNet extends ImageClassifier {

  /**
   * The ResNet requires additional normalization of the used input.
   */
  private static final float IMAGE_MEAN_RED = 123.58f;
  private static final float IMAGE_MEAN_GREEN = 116.779f;
  private static final float IMAGE_MEAN_BLUE = 103.939f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
   * This isn't part of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray = null;

  /**
   * Initializes an {@code ImageClassifier}.
   *
   * @param activity
   */
  ImageClassifierFloatResNet(Activity activity) throws IOException {
    super(activity);
    labelProbArray = new float[1][getNumLabels()];
  }

  @Override
  protected String getModelPath() {
    return "model_transformed.tflite";
  }

  @Override
  protected String getLabelPath() {
    return "labels_imagenet_slim.txt";
  }

  @Override
  protected int getImageSizeX() {
    return 224;
  }

  @Override
  protected int getImageSizeY() {
    return 224;
  }

  @Override
  protected int getNumBytesPerChannel() {
    // a 32bit float value requires 4 bytes
    return 4;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat(((pixelValue >> 16) & 0xFF) - IMAGE_MEAN_RED);
    imgData.putFloat(((pixelValue >> 8) & 0xFF) - IMAGE_MEAN_GREEN);
    imgData.putFloat((pixelValue & 0xFF) - IMAGE_MEAN_BLUE);
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    // TODO the following value isn't in [0,1] yet, but may be greater. Why?
    return getProbability(labelIndex);
  }

  @Override
  protected void runInference() {
    tflite.run(imgData, labelProbArray);
  }
}
```

After that, you need to change the image classifier class used in `Camera2BasicFragment.java`. Locate the function named `onActivityCreated` and change its content as below. Now you will be able to use the compressed ResNet-18 model to classify objects on your mobile phone in real time.

``` Java
/** Load the model and labels. */
@Override
public void onActivityCreated(Bundle savedInstanceState) {
  super.onActivityCreated(savedInstanceState);
  try {
    classifier = new ImageClassifierFloatResNet(getActivity());
  } catch (IOException e) {
    Log.e(TAG, "Failed to initialize an image classifier.", e);
  }
  startBackgroundThread();
}
```
