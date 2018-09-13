'''
Desc: result analysis of training model.
Author: Kris Peng
Data: https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html
Data Desc: 1850/01~2015/12; monthly SST
           1.0 degree latitude x 1.0 degree longitude global grid (360x180).
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy.io as sio
import os.path
from matplotlib import pyplot
from netCDF4 import Dataset
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

# seq = Sequential()
# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    input_shape=(None, 10, 50, 1),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))

# seq.compile(loss='mse', optimizer='adadelta')
# print(seq.summary())

MAX = 31.18499947
MIN = 20.33499908
MEAN = 26.80007865

def normalization(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = (data[i][j]- MIN)/(MAX - MIN)
    return data

def inverse_normalization(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = data[i][j]*(MAX - MIN) + MIN
    return data

# data preprocessing
# sst = '../../../../dataset/sst_grid_1/convert_sst.mon.mean_185001_201512.mat'
sst = '../../data/sst_grid/convert_sst.mon.mean_1850_01_2015_12.mat'

sst_data = sio.loadmat(sst)
sst_data = sst_data['sst'][:,:,:]
sst_data = np.array(sst_data, dtype=float)
# (180 * 360 * 2004) --> (10 * 50 * 2004) NINO3.4 region (5W~5N, 170W~120W)
sst_data = sst_data[85:95,190:240,:]

# todo
# # 1850.01~2015.01 (train)
# train_data = sst_data[::,::,0:-12]
# # 2015.01~2015.12 (test)
# test_data = sst_data[::,::,-12:]

# sst min:20.33 / max:31.18
convert_sst = np.zeros((167,12,10,50,1), dtype = np.float)
for i in range(167):
    for k in range(12):
        # Year * 12 + currentMonth
        convert_sst[i,k,::,::,0] = normalization(sst_data[::,::,i*12+k])

sst_grid = convert_sst

# fit model
file_path = '../../record/grid_pattern/models/40000withNormalization/40000epoch.h5'
file_path = '40000epoch.h5'

seq = load_model(file_path)

which_year = 166
track = sst_grid[which_year][:7, ::, ::, ::]

for j in range(12):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

# to the ground truth
track2 = sst_grid[which_year][::, ::, ::, ::]
for i in range(12):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)
    if i >= 7:
        ax.text(1, 3, 'Predictions', fontsize=12, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=12)
    toplot = inverse_normalization(track[i, ::, ::, 0])
    plt.imshow(toplot)
    cbar = plt.colorbar(plt.imshow(toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)

    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=12)
    toplot = inverse_normalization(track2[i, ::, ::, 0])
    plt.imshow(toplot)
    cbar = plt.colorbar(plt.imshow(toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)

    plt.savefig('%i_animate.png' % (i + 1))
