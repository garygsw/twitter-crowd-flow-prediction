# crowd-flow-prediction
[Crowd Flow Prediction](https://github.com/garygsw/crowd-flow-prediction): A **Deep Learning** Toolbox for Spatio-Temporal Data with additional features to include text information from the internet.

*Tested on `Linux (Fedora 23 and Ubuntu 14.04LTS)`.*

## Installation

This project rely on the DeepST library, which uses the following dependencies:

* [Keras](https://keras.io/#installation) and its dependencies are required to use DeepST. Please read [Keras Configuration](keras_configuration.md) for the configuration setting.
* [Theano](http://deeplearning.net/software/theano/install.html#install) or [TensorFlow](https://github.com/tensorflow/tensorflow#download-and-setup), but **Tensorflow** is recommended.
* numpy and scipy
* HDF5 and [h5py](http://www.h5py.org/)
* [pandas](http://pandas.pydata.org/)
* CUDA 7.5 or latest version. And **cuDNN** is highly recommended.

To install DeepST, `cd` to the **DeepST** folder and run the install command:

```
python setup.py install
```

## License

DeepST is released under the MIT License (refer to the LICENSE file for details).
