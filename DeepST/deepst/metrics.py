import numpy as np
from keras import backend as K


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# aliases
mse = MSE = mean_squared_error
# rmse = RMSE = root_mean_square_error

# def masked_mean_squared_error(y_true, y_pred):
#     idx = (y_true < 2).nonzero()   # get valid index
#     return K.mean(K.square(y_pred[idx] - y_true[idx]))


def masked_rmse(mask):
    if K.backend() == 'tensorflow':
        def masked_rmse(y_true, y_pred):
            idx = K.expand_dims(mask, axis=0)
            idx = K.tile(idx, [K.shape(y_true)[0], 1, 1, 1])
            idx = K.cast(idx, 'float32')
            zero = K.equal(idx, K.constant(1, dtype='float32'))
            zero = K.cast(zero, 'float32')
            return (K.sum(K.square(y_pred - y_true) * zero) / K.sum(idx)) ** 0.5
    elif K.backend() == 'theano':
        def masked_rmse(y_true, y_pred):
            idx = K.tile(mask, [y_true.shape[0], 1, 1, 1])
            idx = idx.nonzero()  # to make it a tensor mask
            return K.mean(K.square(y_pred[idx] - y_true[idx])) ** 0.5
    return masked_rmse


def masked_mse(mask):
    if K.backend() == 'tensorflow':
        def masked_mse(y_true, y_pred):
            idx = K.expand_dims(mask, axis=0)
            idx = K.tile(idx, [K.shape(y_true)[0], 1, 1, 1])
            idx = K.cast(idx, 'float32')
            zero = K.equal(idx, K.constant(1, dtype='float32'))
            zero = K.cast(zero, 'float32')
            return K.sum(K.square(y_pred - y_true) * zero) / K.sum(idx)
    elif K.backend() == 'theano':
        def masked_mse(y_true, y_pred):
            idx = K.tile(mask, [y_true.shape[0], 1, 1, 1])
            idx = idx.nonzero()  # to make it a tensor mask
            return K.mean(K.square(y_pred[idx] - y_true[idx]))
    return masked_mse
