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
import ConvLSTM2D
import STResNet
import os.path
from matplotlib import pyplot
from keras.models import load_model
from keras.utils import multi_gpu_model

def CovLSTM2D_model():
    seq = ConvLSTM2D.model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    return seq

def STResNet_model():
    seq = STResNet.model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    return seq

# parameters setting
epochs = 2
batch_size = 10
validation_split = 0.05
train_length = 160
which_year = 166 # year to visulization

def main():
    # data preparation
    sst_grid = pp.load_data_convlstm()
    # sst_grid = pp.load_data_resnet()

    print("Whole Dataset Shape: %s " % str(sst_grid.shape))

    # fit model
    file_path = '20000epoch.h5'

    # model setting
    seq = CovLSTM2D_model()
    # seq = STResNet_model()


    seq.compile(loss='mse', optimizer='adadelta')

    if not os.path.exists(file_path):
        # ConLSTM
        history = seq.fit(sst_grid[:train_length], sst_grid[:train_length], batch_size=batch_size, epochs=epochs, validation_split=validation_split)

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
    # feed it with the first 6 patterns
    # predict the new patterns
    pred_sequence = sst_grid[which_year][:6, ::, ::, ::]

    for j in range(12):
        new_pos = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
        new = new_pos[::, -1, ::, ::, ::]
        pred_sequence = np.concatenate((pred_sequence, new), axis=0)

    # And then compare the predictions with the ground truth
    act_sequence = sst_grid[which_year][::, ::, ::, ::]

    for i in range(12):
        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(311)
        if i >= 6:
            ax.text(1, 3, 'Prediction', fontsize=12)
        else:
            ax.text(1, 3, 'Initial trajectory', fontsize=12)
        toplot1 = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
        plt.imshow(toplot1)
        cbar = plt.colorbar(plt.imshow(toplot1), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        ax = fig.add_subplot(312)
        plt.text(1, 3, 'Ground truth', fontsize=12)
        toplot2 = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
        plt.imshow(toplot2)
        cbar = plt.colorbar(plt.imshow(toplot2), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        ax = fig.add_subplot(313)
        plt.text(1, 3, 'Difference', fontsize=12)
        toplot3 = toplot2 - toplot1
        plt.imshow(toplot3)
        cbar = plt.colorbar(plt.imshow(toplot3), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        plt.savefig('%i_animate.png' % (i + 1))

if __name__ == '__main__':
    main()
