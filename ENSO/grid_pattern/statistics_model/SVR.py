'''
Desc: the SVR model of grid pattern.
Author: Kris Peng
Data: https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html
Data Desc: 1850/01~2015/12; monthly SST
           1.0 degree latitude x 1.0 degree longitude global grid (360x180).
           for NINO34 region (50*10)
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import sys
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import pandas as pd
from math import sqrt

sys.setrecursionlimit(100000000)

# monthly sst parameters setting
train_length = 500
len_seq = 1980
len_frame = 12
start_seq = 1801
end_seq = 1968
height, width = 10, 50
MAX = 31.18499947
MIN = 20.33499908
MEAN = 26.80007865

DATA_PATH = 'monthly_sst+1.npy'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def normalization(data):
    normalized_data = np.zeros(len_frame, dtype=np.float)
    for i in range(len(data)):
            normalized_data[i] = (data[i]- MIN)/(MAX - MIN)
    return normalized_data

def inverse_normalization(data):
    inverse_data = np.zeros(len_frame, dtype=np.float)
    for i in range(len(data)):
            inverse_data[i] = data[i]*(MAX - MIN) + MIN
    return inverse_data

def main():
    log_file_path = "SVR.log"
    log = open(log_file_path,'w')

    train_X_raw, train_Y_raw, sst_grid = np.load(DATA_PATH)

    # SVR Model -n ahead
    train_X = np.zeros((train_length*height*width, len_frame), dtype=np.float)
    train_Y = np.zeros((train_length*height*width, 1), dtype=np.float)

    for i in range(train_length):
        for m in range(width):
            for n in range(height):
                # print(sst_grid[i,0:len_frame,n,m,0])
                train_X[m+n,::] = normalization(sst_grid[i,0:len_frame,n,m,0])
                train_Y[m+n,0] = (sst_grid[i+1,-1,n,m,0]- MIN)/(MAX - MIN)

    model_sum_rmse = 0
    base_sum_rmse = 0
    model_sum_mae = 0
    base_sum_mae = 0
    model_sum_mape = 0
    base_sum_mape = 0

    clf = SVR()
    clf.fit(train_X, train_Y)

    for k in range(start_seq, end_seq):
        print(k)
        # rolling-forecasting with -n steps
        pred_sequence_raw = sst_grid[k][0:len_frame, ::, ::, ::]
        pred_sequence = sst_grid[k][0:len_frame, ::, ::, ::]
        act_sequence = sst_grid[k+len_frame][0:len_frame, ::, ::, ::]
        for j in range(len_frame):
            pred_toplot = np.zeros((1,10,50,1), dtype=np.float)
            # single point prediction
            for i in range(width):
                for m in range(height):
                    history = np.array([float(x) for x in pred_sequence[::,m,i,::]])
                    history = normalization(history.flatten())
                    history = history.reshape(1, len_frame)
                    output = clf.predict((history))
                    pred_toplot[0][m][i][0] = output*(MAX-MIN)+MIN
            pred_sequence = np.concatenate((pred_sequence, pred_toplot), axis=0)
            pred_toplot = pred_toplot[0,::,::,0]

            # abandon the first frame
            pred_sequence = pred_sequence[1:len_frame+1, ::, ::, ::]
            baseline_frame = pred_sequence_raw[j, ::, ::, 0]
            act_toplot = act_sequence[j, ::, ::, 0]

            model_rmse = mean_squared_error(act_toplot, pred_toplot)
            baseline_rmse = mean_squared_error(act_toplot, baseline_frame)

            model_mae = mean_absolute_error(act_toplot, pred_toplot)
            baseline_mae = mean_absolute_error(act_toplot, baseline_frame)

            model_mape = mean_absolute_percentage_error(act_toplot, pred_toplot)
            baseline_mape = mean_absolute_percentage_error(act_toplot, baseline_frame)

            model_sum_rmse, base_sum_rmse = model_sum_rmse + model_rmse, base_sum_rmse + baseline_rmse
            model_sum_mae, base_sum_mae = model_sum_mae + model_mae, base_sum_mae + baseline_mae
            model_sum_mape, base_sum_mape = model_sum_mape + model_mape, base_sum_mape + baseline_mape

        log.write("\n ============")
        log.write("\n Round: %s" % k)
        log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(k-start_seq+1)))))
        log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(k-start_seq+1)))))
        log.write("\nTotal Model MAE: %s" % (model_sum_mae/(len_frame*(k-start_seq+1))))
        log.write("\nTotal Baseline MAE: %s" % (base_sum_mae/(len_frame*(k-start_seq+1))))
        log.write("\nModel MAPE: %s" % (model_sum_mape/(len_frame*(k-start_seq+1))))
        log.write("\nBaseline MAPE: %s" % (base_sum_mape/(len_frame*(k-start_seq+1))))

        print("Total Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(k-start_seq+1)))))
        print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(k-start_seq+1)))))
        print("Total Model MAE: %s" % (model_sum_mae/(len_frame*(k-start_seq+1))))
        print("Total Baseline MAE: %s" % (base_sum_mae/(len_frame*(k-start_seq+1))))
        print("Model MAPE: %s" % (model_sum_mape/(len_frame*(k-start_seq+1))))
        print("Baseline MAPE: %s" % (base_sum_mape/(len_frame*(k-start_seq+1))))

        if k%40 == 0:
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(321)
            ax.text(1, 3, 'Prediction', fontsize=12)
            plt.imshow(pred_toplot)
            cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
            cbar.set_label('°C',fontsize=12)

            # 将预测seq-n数据作为baseline
            ax = fig.add_subplot(322)
            plt.text(1, 3, 'Baseline', fontsize=12)
            plt.imshow(baseline_frame)
            cbar = plt.colorbar(plt.imshow(baseline_frame), orientation='horizontal')
            cbar.set_label('°C',fontsize=12)

            ax = fig.add_subplot(323)
            plt.text(1, 3, 'Ground truth', fontsize=12)
            plt.imshow(act_toplot)
            cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
            cbar.set_label('°C',fontsize=12)

            ax = fig.add_subplot(324)
            plt.text(1, 3, 'Ground truth', fontsize=12)
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
            plt.savefig('%i_%i_animate.png' % ((k + 1), (j + 1)))

    print("="*10)
    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Model MAE: %s" % (model_sum_mae/(len_frame*(end_seq-start_seq))))
    print("Total Baseline MAE: %s" % (base_sum_mae/(len_frame*(end_seq-start_seq))))
    print("Model MAPE: %s" % (model_sum_mape/(len_frame*(end_seq-start_seq))))
    print("Baseline MAPE: %s" % (base_sum_mape/(len_frame*(end_seq-start_seq))))

    log.write("\n ============")
    log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(end_seq-start_seq)))))
    log.write("\nTotal Model MAE: %s" % (model_sum_mae/(len_frame*(end_seq-start_seq))))
    log.write("\nTotal Baseline MAE: %s" % (base_sum_mae/(len_frame*(end_seq-start_seq))))
    log.write("\nModel MAPE: %s" % (model_sum_mape/(len_frame*(end_seq-start_seq))))
    log.write("\nBaseline MAPE: %s" % (base_sum_mape/(len_frame*(end_seq-start_seq))))
    log.close()

if __name__ == '__main__':
    main()
