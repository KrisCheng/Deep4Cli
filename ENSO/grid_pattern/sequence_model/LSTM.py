'''
Desc: LSTM Model.
Author: Kris Peng
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers import Dense,Embedding,TimeDistributed
from keras.layers import LSTM, Flatten, LSTMCell
from keras.models import load_model

height, width = 10, 50

def model():
    # Model
    seq = Sequential()
    seq.add(Dense(units = 1000, input_shape=(None, height, width, 1)))
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    return seq
