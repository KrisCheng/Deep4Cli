'''
Desc: Spatio Temporal ResNet Model.
Author: Kris Peng
'''
from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    add,
    Dense,
    Reshape
)
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
#from keras.utils.visualize_util import plot

def _shortcut(input, residual):
    return add([input, residual])

def _bn_relu_conv(filters, nb_row, nb_col, subsample=(1, 1), bn=True):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(filters=filters, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
    return f


def _residual_unit(filters, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(filters, 3, 3)(input)
        residual = _bn_relu_conv(filters, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, filters, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(filters=filters,
                                  init_subsample=init_subsample)(input)
        return input
    return f


def model(conf=(12, 1, 10, 50), nb_residual_unit=1):

    '''
    C - Temporal Closeness
    todo P - Period
    todo T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    '''
    # main input
    main_inputs = []
    outputs = []

    if conf is not None:
        len_seq, nb_flow, map_height, map_width = conf
        input = Input(shape=(nb_flow * len_seq, map_height, map_width))
        main_inputs.append(input)
        # Conv1
        conv1 = Conv2D(
            filters=64, nb_row=3, nb_col=3, padding='same')(input)

        # [nb_residual_unit] Residual Units
        residual_output = ResUnits(_residual_unit, filters=64,
                          repetations=nb_residual_unit)(conv1)

        # Conv2
        activation = Activation('relu')(residual_output)
        conv2 = Conv2D(
            filters=nb_flow, nb_row=3, nb_col=3, padding='same')(activation)

        outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from .iLayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)
    return model
