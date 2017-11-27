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
    def masked_rmse(y_true, y_pred):
        size = K.int_shape(y_true)[0]  # get len of timeslots
        if size is not None:
            idx = np.tile(mask, [size, 1, 1, 1])
        else:
            idx = y_true
        idx = idx.nonzero()  # to make it a tensor variable
        return K.mean(K.square(y_pred[idx] - y_true[idx])) ** 0.5
    return masked_rmse
