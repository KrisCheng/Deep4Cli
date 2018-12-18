  '''
Desc: CNN Model.
Author: Kris Peng
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

height, width = 10, 50

def model():
    # Model
    seq = Sequential()
    seq.add(Dense(units = 50, input_shape=(None, height, width, 1)))
    seq.add(Conv3D(filters=40, kernel_size=(3, 3, 3),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    seq.add(Conv3D(filters=40, kernel_size=(3, 3, 3),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    return seq
