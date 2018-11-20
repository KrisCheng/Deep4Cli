'''
Desc: analysis of the generated pattern.
Author: Kris Peng
Data: https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html
Data Desc: 1850/01~2015/12; monthly SST
           1.0 degree latitude x 1.0 degree longitude global grid (360x180).
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import matplotlib as mpl
import os.path
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

MAX = 31.18499947
MIN = 20.33499908
MEAN = 26.80007865
len_frame = 12
len_seq = 1980
map_height, map_width = 10, 50
fold_name = "SST_Pattern"
DATA_PATH = "monthly_sst+1.npy"

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

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_ture))

def main():
    os.makedirs(fold_name)
    file_path = "model_2000_epochs_12.h5"
    log_file_path = fold_name+'/record.log'
    log = open(log_file_path, 'w')

    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file

    # normalization, data for ConvLSTM Model -n ahead -5 dimension
    train_X = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    train_Y = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    sst_grid = np.zeros((len_seq+len_frame, len_frame, 10, 50, 1), dtype=np.float)
    for i in range(len_seq):
        for k in range(len_frame):
            train_X[i,k,::,::,0] = normalization(train_X_raw[i,k,::,::,0])
            train_Y[i,k,::,::,0] = normalization(train_Y_raw[i,k,::,::,0])
            sst_grid[i,k,::,::,0] = normalization(sst_grid_raw[i,k,::,::,0])

    seq = load_model(file_path)

    model_sum_loss = 0
    base_sum_loss = 0

    # 2015.01
    pred_sequence_raw = sst_grid[-12][::, ::, ::, ::]
    pred_sequence = sst_grid[-12][::, ::, ::, ::]
    act_sequence = sst_grid[-1][::, ::, ::, ::]

    model_sum_rmse = 0
    base_sum_rmse = 0
    model_sum_mae = 0
    base_sum_mae = 0
    model_sum_mape = 0
    base_sum_mape = 0

    for j in range(len_frame):
        new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
        new = new_frame[::, -1, ::, ::, ::]
        pred_sequence = np.concatenate((pred_sequence, new), axis=0)

        baseline_frame = inverse_normalization(pred_sequence_raw[j, ::, ::, 0])
        pred_toplot = inverse_normalization(pred_sequence[-1, ::, ::, 0])
        act_toplot = inverse_normalization(act_sequence[j, ::, ::, 0])

        pred_sequence = pred_sequence[1:len_frame+1, ::, ::, ::]

        model_rmse = mean_squared_error(act_toplot, pred_toplot)
        baseline_rmse = mean_squared_error(act_toplot, baseline_frame)

        model_mae = mean_absolute_error(act_toplot, pred_toplot)
        baseline_mae = mean_absolute_error(act_toplot, baseline_frame)

        model_mape = mean_absolute_percentage_error(act_toplot, pred_toplot)
        baseline_mape = mean_absolute_percentage_error(act_toplot, baseline_frame)

        model_sum_rmse, base_sum_rmse = model_sum_rmse + model_rmse, base_sum_rmse + baseline_rmse
        model_sum_mae, base_sum_mae = model_sum_mae + model_mae, base_sum_mae + baseline_mae
        model_sum_mape, base_sum_mape = model_sum_mape + model_mape, base_sum_mape + baseline_mape
   
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(321)
        ax.text(1, 3, 'Prediction', fontsize=12)
        pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
        plt.imshow(pred_toplot)
        cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        # 将预测seq-12数据作为baseline
        baseline_frame = pp.inverse_normalization(pred_sequence_raw[i, ::, ::, 0])
        ax = fig.add_subplot(322)
        plt.text(1, 3, 'Baseline', fontsize=12)
        plt.imshow(baseline_frame)
        cbar = plt.colorbar(plt.imshow(baseline_frame), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        ax = fig.add_subplot(323)
        plt.text(1, 3, 'Ground truth', fontsize=12)
        act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
        plt.imshow(act_toplot)
        cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        ax = fig.add_subplot(324)
        plt.text(1, 3, 'Ground truth', fontsize=12)
        act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
        plt.imshow(act_toplot)
        cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        ax = fig.add_subplot(325)
        plt.text(1, 3, 'Diff_Pred', fontsize=12)
        diff_toplot = act_toplot - pred_toplot
        plt.imshow(diff_toplot)
        cbar = plt.colorbar(plt.imshow(diff_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        ax = fig.add_subplot(326)
        plt.text(1, 3, 'Diff_Base', fontsize=12)
        diff_toplot = act_toplot - baseline_frame
        plt.imshow(diff_toplot)
        cbar = plt.colorbar(plt.imshow(diff_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)
        plt.savefig(fold_name + '/%s_animate.png' % (str(j+1)))
            
    log.write("\n\n ============")
    log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse/len_frame)))
    log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse/len_frame)))
    log.write("\nTotal Model MAE: %s" % (model_sum_mae/len_frame))
    log.write("\nTotal Baseline MAE: %s" % (base_sum_mae/len_frame))
    log.write("\nModel MAPE: %s" % (model_sum_mape/len_frame))
    log.write("\nBaseline MAPE: %s" % (base_sum_mape/len_frame))  

    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse/len_frame)))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse/len_frame)))
    print("Total Model MAE: %s" % (model_sum_mae/len_frame))
    print("Total Baseline MAE: %s" % (base_sum_mae/len_frame))
    print("Model MAPE: %s" % (model_sum_mape/len_frame))
    print("Baseline MAPE: %s" % (base_sum_mape/len_frame))

if __name__ == '__main__':
    main()
