'''
Desc: the ARIMA model of grid pattern.
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
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import pandas as pd
from math import sqrt

sys.setrecursionlimit(100000000)

# monthly sst parameters setting
train_length = 1800
len_seq = 1980
len_frame = 9
start_seq = 1801
end_seq = 1968
MEAN = 26.80007865

DATA_PATH = 'monthly_sst+1.npy'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def main():
    log_file_path = "ARIMA.log"
    log = open(log_file_path,'w')

    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH)  # from .npy file

    # ARIMA Model -12 ahead
    train_X = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    train_Y = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    sst_grid = np.zeros((len_seq+9, len_frame, 10, 50, 1), dtype=np.float)
    for i in range(len_seq):
        for k in range(len_frame):
            train_X[i,k,::,::,0] = train_X_raw[i,k,::,::,0]
            train_Y[i,k,::,::,0] = train_Y_raw[i,k,::,::,0]
            sst_grid[i,k,::,::,0] = sst_grid_raw[i,k,::,::,0]

    for m in range(len_frame):
        sst_grid[len_seq,m,::,::,0] = sst_grid_raw[len_seq,m,::,::,0]

    model_sum_rmse_12 = 0
    base_sum_rmse_12 = 0
    model_sum_mae_12 = 0
    base_sum_mae_12 = 0
    model_sum_mape_12 = 0
    base_sum_mape_12 = 0

    for k in range(start_seq, end_seq):
        print(k)
        # rolling-forecasting with -12 steps
        pred_sequence_raw = sst_grid[k][::, ::, ::, ::]
        pred_sequence = sst_grid[k][::, ::, ::, ::]
        act_sequence = sst_grid[k+len_frame][::, ::, ::, ::]
        for j in range(len_frame):
            pred_toplot = np.zeros((1,10,50,1), dtype=np.float)
            # single point prediction
            for i in range(50):
                for m in range(10):
                    history = np.array([float(x) for x in pred_sequence[::,m,i,::]])
                    history = pd.Series(history)
                    where_are_nan = np.isnan(history)
                    where_are_inf = np.isinf(history)
                    history[where_are_nan] = MEAN
                    history[where_are_inf] = MEAN
                    model = ARIMA(history, order=(0,0,0))
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast()
                    pred_toplot[0][m][i][0] = output[0]
            pred_sequence = np.concatenate((pred_sequence, pred_toplot), axis=0)
            pred_toplot = pred_toplot[0,::,::,0]
            where_are_nan = np.isnan(pred_toplot)
            where_are_inf = np.isinf(pred_toplot)
            pred_toplot[where_are_nan] = MEAN
            pred_toplot[where_are_inf] = MEAN

            # log.write(str(pred_toplot))
            # abandon the first frame
            pred_sequence = pred_sequence[1:7, ::, ::, ::]
            baseline_frame = pred_sequence_raw[j, ::, ::, 0]
            act_toplot = act_sequence[j, ::, ::, 0]

            model_rmse = mean_squared_error(act_toplot, pred_toplot)
            baseline_rmse = mean_squared_error(act_toplot, baseline_frame)

            model_mae = mean_absolute_error(act_toplot, pred_toplot)
            baseline_mae = mean_absolute_error(act_toplot, baseline_frame)

            model_mape = mean_absolute_percentage_error(act_toplot, pred_toplot)
            baseline_mape = mean_absolute_percentage_error(act_toplot, baseline_frame)

            model_sum_rmse_12, base_sum_rmse_12 = model_sum_rmse_12 + model_rmse, base_sum_rmse_12 + baseline_rmse
            model_sum_mae_12, base_sum_mae_12 = model_sum_mae_12 + model_mae, base_sum_mae_12 + baseline_mae
            model_sum_mape_12, base_sum_mape_12 = model_sum_mape_12 + model_mape, base_sum_mape_12 + baseline_mape

        log.write("\n ============")
        log.write("\n Round: %s" % k)
        log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse_12/(len_frame*(k-start_seq+1)))))
        log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse_12/(len_frame*(k-start_seq+1)))))
        log.write("\nTotal Model MAE: %s" % (model_sum_mae_12/(len_frame*(k-start_seq+1))))
        log.write("\nTotal Baseline MAE: %s" % (base_sum_mae_12/(len_frame*(k-start_seq+1))))
        log.write("\nModel MAPE: %s" % (model_sum_mape_12/(len_frame*(k-start_seq+1))))
        log.write("\nBaseline MAPE: %s" % (base_sum_mape_12/(len_frame*(k-start_seq+1))))

        print("Total Model RMSE: %s" % (sqrt(model_sum_rmse_12/(len_frame*(k-start_seq+1)))))
        print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse_12/(len_frame*(k-start_seq+1)))))
        print("Total Model MAE: %s" % (model_sum_mae_12/(len_frame*(k-start_seq+1))))
        print("Total Baseline MAE: %s" % (base_sum_mae_12/(len_frame*(k-start_seq+1))))
        print("Model MAPE: %s" % (model_sum_mape_12/(len_frame*(k-start_seq+1))))
        print("Baseline MAPE: %s" % (base_sum_mape_12/(len_frame*(k-start_seq+1))))

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(321)
        ax.text(1, 3, 'Prediction', fontsize=12)
        plt.imshow(pred_toplot)
        cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)

        # 将预测seq-12数据作为baseline
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
    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse_12/(len_frame*(end_seq-start_seq)))))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse_12/(len_frame*(end_seq-start_seq)))))
    print("Total Model MAE: %s" % (model_sum_mae_12/(len_frame*(end_seq-start_seq))))
    print("Total Baseline MAE: %s" % (base_sum_mae_12/(len_frame*(end_seq-start_seq))))
    print("Model MAPE: %s" % (model_sum_mape_12/(len_frame*(end_seq-start_seq))))
    print("Baseline MAPE: %s" % (base_sum_mape_12/(len_frame*(end_seq-start_seq))))
    log.write("\n ============")
    log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse_12/(len_frame*(end_seq-start_seq)))))
    log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse_12/(len_frame*(end_seq-start_seq)))))
    log.write("\nTotal Model MAE: %s" % (model_sum_mae_12/(len_frame*(end_seq-start_seq))))
    log.write("\nTotal Baseline MAE: %s" % (base_sum_mae_12/(len_frame*(end_seq-start_seq))))
    log.write("\nModel MAPE: %s" % (model_sum_mape_12/(len_frame*(end_seq-start_seq))))
    log.write("\nBaseline MAPE: %s" % (base_sum_mape_12/(len_frame*(end_seq-start_seq))))
    log.close()

if __name__ == '__main__':
    main()
