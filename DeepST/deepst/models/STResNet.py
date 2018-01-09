'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetitions=1):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(map_height, map_width, len_closeness, len_period, len_trend,
             external_dim, nb_filters=64, kernal_size=(3, 3),
             nb_residual_unit=2, use_tweet_counts=False):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    '''

    kernal_w, kernal_h = kernal_size
    if use_tweet_counts:
        input_dim = 3
    else:
        input_dim = 2

    # main input
    main_inputs = []
    outputs = []
    for len_seq in [len_closeness, len_period, len_trend]:
        if len_seq is not None:
            input = Input(shape=(input_dim * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            conv1 = Convolution2D(nb_filter=nb_filters,
                                  nb_row=kernal_h,
                                  nb_col=kernal_w,
                                  border_mode="same")(input)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit,
                                       nb_filter=nb_filters,
                                       repetitions=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution2D(nb_filter=2,  # output dim of prediction
                                  nb_row=kernal_h,
                                  nb_col=kernal_w,
                                  border_mode="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from .iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = merge(new_outputs, mode='sum')

    # fusing with external component
    if external_dim is not None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(output_dim=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(output_dim=2 * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((2, map_height, map_width))(activation)
        main_output = merge([main_output, external_output], mode='sum')
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model

if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
