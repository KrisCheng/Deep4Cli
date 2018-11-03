'''
Desc: result analysis of the final model.
Author: Kris Peng
Data: https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html
Data Desc: 1850/01~2015/12; monthly SST
           1.0 degree latitude x 1.0 degree longitude global grid (360x180).
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import matplotlib as mpl
import os
import os.path
from matplotlib import pyplot
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt

MAX = 31.18499947
MIN = 20.33499908
MEAN = 26.80007865
train_length = 1800
len_seq = 1980
start_seq = 1801
begin_pred_seq = 12
end_pred_seq = 24
DATA_PATH = '../../../../dataset/sst_grid_1/sst_monthly_185001_201512.npy'
fold_name = 'imgs'
file_path = "model_2000_epochs.h5"
map_height, map_width = 10, 50

# 0~1 Normalization
def normalization(data):
    normalized_data = np.zeros((map_height, map_width), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            normalized_data[i][j] = (data[i][j]- MIN)/(MAX - MIN)
    return normalized_data

def inverse_normalization(data):
    inverse_data = np.zeros((map_height, map_width), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            inverse_data[i][j] = data[i][j]*(MAX - MIN) + MIN
    return inverse_data

def main():

    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file
    # normalization
    train_X = np.zeros((1980, 24, 10, 50, 1), dtype=np.float)
    train_Y = np.zeros((1980, 24, 10, 50, 1), dtype=np.float)
    sst_grid = np.zeros((1981, 24, 10, 50, 1), dtype=np.float)
    for i in range(1980):
        for k in range(24):
            train_X[i,k,::,::,0] = normalization(train_X_raw[i,k,::,::,0])
            train_Y[i,k,::,::,0] = normalization(train_Y_raw[i,k,::,::,0])
            sst_grid[i,k,::,::,0] = normalization(sst_grid_raw[i,k,::,::,0])
    for m in range(24):
        sst_grid[1980,m,::,::,0] = normalization(sst_grid_raw[1980,m,::,::,0])

    seq = load_model(file_path)
    # os.makedirs(fold_name)

    model_sum_loss = 0
    base_sum_loss = 0

    for k in range(start_seq, len_seq):
        # print(k)
        act_sequence = sst_grid[k][::, ::, ::, ::]
        pred_sequence = sst_grid[k][:12, ::, ::, ::]
        print(pred_sequence[np.newaxis, ::, ::, ::, ::].shape)
        for j in range(12):
            new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
            # TODO why?? ref: keras conv_lstm.py demo
            # print(new_frame.shape)
            new = new_frame[::, -1, ::, ::, ::]
            pred_sequence = np.concatenate((pred_sequence, new), axis=0)
        for i in range(begin_pred_seq, end_pred_seq):
            # fig = plt.figure(figsize=(16, 8))
            # ax = fig.add_subplot(321)
            # ax.text(1, 3, 'Prediction', fontsize=12)
            pred_toplot = inverse_normalization(pred_sequence[i, ::, ::, 0])
            # plt.imshow(pred_toplot)
            # cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
            # cbar.set_label('°C',fontsize=12)
            # # 将预测seq-12数据作为baseline
            baseline_frame = inverse_normalization(act_sequence[11, ::, ::, 0])
            # ax = fig.add_subplot(322)
            # plt.text(1, 3, 'Baseline', fontsize=12)
            # plt.imshow(baseline_frame)
            # cbar = plt.colorbar(plt.imshow(baseline_frame), orientation='horizontal')
            # cbar.set_label('°C',fontsize=12)
            # ax = fig.add_subplot(323)
            # plt.text(1, 3, 'Ground truth', fontsize=12)
            # act_toplot = inverse_normalization(act_sequence[i, ::, ::, 0])
            # plt.imshow(act_toplot)
            # cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
            # cbar.set_label('°C',fontsize=12)
            # ax = fig.add_subplot(324)
            # plt.text(1, 3, 'Ground truth', fontsize=12)
            act_toplot = inverse_normalization(act_sequence[i, ::, ::, 0])
            # plt.imshow(act_toplot)
            # cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
            # cbar.set_label('°C',fontsize=12)
            # ax = fig.add_subplot(325)
            # plt.text(1, 3, 'Diff_Pred', fontsize=12)
            # diff_toplot = act_toplot - pred_toplot
            # plt.imshow(diff_toplot)
            # cbar = plt.colorbar(plt.imshow(diff_toplot), orientation='horizontal')
            # cbar.set_label('°C',fontsize=12)
            # ax = fig.add_subplot(326)
            # plt.text(1, 3, 'Diff_Base', fontsize=12)
            # diff_toplot = act_toplot - baseline_frame
            # plt.imshow(diff_toplot)
            # cbar = plt.colorbar(plt.imshow(diff_toplot), orientation='horizontal')
            # cbar.set_label('°C',fontsize=12)
            # # plt.savefig(fold_name + '/%i_%i_animate.png' % ( (k+1), (i + 1)))

            model_rmse = sqrt(mean_squared_error(act_toplot, pred_toplot))
            baseline_rmse = sqrt(mean_squared_error(act_toplot, baseline_frame))
            model_sum_loss, base_sum_loss = model_sum_loss + model_rmse, base_sum_loss + baseline_rmse

    print("="*10)
    print("Total Model RMSE: %s" % (model_sum_loss/(12*(len_seq-start_seq))))
    print("Total Baseline RMSE: %s" % (base_sum_loss/(12*(len_seq-start_seq))))

if __name__ == '__main__':
    main()
