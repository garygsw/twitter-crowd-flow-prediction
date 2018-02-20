# Keras Configuration

The default *backend* is *tensorflow* and *image_dim_ordering* is *tf*.
Change them accordingly to you own choice of backend.

1. `Keras + TensorFlow` configuration file of Keras: ~/.keras/keras.json

```
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_data_format": "channels_last",
    "backend": "tensorflow"
}
```

2. `Keras + Theano` configuration file of Keras: ~/.keras/keras.json
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```
Ref: <https://keras.io/backend/>


# Keras Library Tweaks

The following changes are required to be performed to the Keras native library if tweets' tokens index input is used in the model. Hence, it is **highly recommended** to create a virtual environment, and edit the Keras library in the environment instead. See [conda envs](https://conda.io/docs/using/envs.html) for instructions to create one.

1. For Keras 2.1.3 to be compatible with Tensorflow 0.12.1 as backend, the following change to the library is required **(not required for Tensorflow 1.0.1)**:

*@ keras/backend/tensorflow_backend.py: line 1878*
change:
```
return tf.concat([to_dense(x) for x in tensors], axis)
```
to:
```
return tf.concat(axis, [to_dense(x) for x in tensors])
```

2. If input format for tweets' tokens index is sparse, the following changes to the Keras library is required:

(a) add sparse matrices shape check during model compilation:
*@ keras/engine/training.py: line 106-108*
change:
```
data_shape = data[i].shape
shape = shapes[i]
if data[i].ndim != len(shape):
```
to:
```
if hasattr(data[i][0][0], 'indices')  # if it is a sparse matrix
    data_shape = (data[i].shape[0], ) + data[i][0][0].shape
else:
    data_shape = data[i].shape
shape = shapes[i]
if len(data_shape) != len(shape):
```

(b) combine each batch of sparse matrices:
*@ keras/backend/tensorflow_backend.py: at the top (imports section)*
add:
```
import scipy.sparse
```
*@ keras/backend/tensorflow_backend.py: line 2466-2467*
change:
```
if is_sparse(tensor):
    sparse_coo = value.tocoo()
```
to:
```
if is_sparse(tensor):
    value = scipy.sparse.vstack([x[0] for x in value])
    sparse_coo = value.tocoo()
```

# Troubleshooting

In our experiment, we mainly tested on `Keras 2.1.3` with backend `Theano 0.9.0` and `Tensorflow 1.0.1 / 0.12.1`.

Sparse inputs currently does not work with Theano backend.
