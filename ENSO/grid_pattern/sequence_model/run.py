'''
Desc: the entrance of the whole project.
Author: Kris Peng
Data: https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html
Data Desc: 1850/01~2015/12; monthly SST
           1.0 degree latitude x 1.0 degree longitude global grid (360x180).
           for NINO34 region (50*10)
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import preprocessing as pp
import ConvLSTM2d
import os.path
from matplotlib import pyplot
from keras.utils import multi_gpu_model

def build_model():
    seq = ConvLSTM2d.model()
    seq = multi_gpu_model(seq, gpus=2)
    seq.compile(loss='mse', optimizer='adadelta')
    print(seq.summary())
    return seq

# parameters setting
epochs = 2
batch_size = 10
validation_split = 0.05
which_year = 166 # year to visulization

sst_grid = pp.load_data()

# fit model
file_path = '40000epoch.h5'

seq = build_model()

if not os.path.exists(file_path):
    history = seq.fit(sst_grid[:160], sst_grid[:160], batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    # seq.save(file_path)
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper left')
    pyplot.show()
else:
    seq = load_model(file_path)

# Testing the network on new monthly SST distribution
# feed it with the first 7 patterns
# predict the new patterns
track = sst_grid[which_year][:6, ::, ::, ::]

for j in range(12):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

# And then compare the predictions
# to the ground truth
track2 = sst_grid[which_year][::, ::, ::, ::]
for i in range(12):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)
    if i >= 6:
        ax.text(1, 3, 'Prediction', fontsize=12, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=12)
    toplot = pp.inverse_normalization(track[i, ::, ::, 0])
    plt.imshow(toplot)
    cbar = plt.colorbar(plt.imshow(toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=12)
    toplot = pp.inverse_normalization(track2[i, ::, ::, 0])
    plt.imshow(toplot)
    cbar = plt.colorbar(plt.imshow(toplot), orientation='horizontal')
    cbar.set_label('°C',fontsize=12)

    # plt.savefig('%i_animate.png' % (i + 1))
