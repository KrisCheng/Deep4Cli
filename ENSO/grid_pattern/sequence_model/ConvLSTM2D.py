'''
Desc: ConvLSTM2D Model.
Author: Kris Peng
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

height, width = 10, 50

def model():
    # Model
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                       input_shape=(None, height, width, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    return seq
