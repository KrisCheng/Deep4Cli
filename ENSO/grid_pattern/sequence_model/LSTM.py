'''
Desc: LSTM Model.
Author: Kris Peng
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

height, width = 10, 50

def model():
    # Model
    seq = Sequential()
    seq.add(LSTM(60,input_shape=(None, height, width, 1), return_sequences=True))
    seq.add(LSTM(60, padding='same', return_sequences=True))
    seq.add(LSTM(60, padding='same', return_sequences=True))
    seq.add(LSTM(60, padding='same', return_sequences=True))
    seq.add(Dense(activation='relu', padding='same', data_format='channels_last'))
    return seq
