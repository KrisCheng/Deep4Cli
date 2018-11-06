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
import FNN
import CNN
import LSTM
import os
import os.path
from matplotlib import pyplot
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from contextlib import redirect_stdout

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

def FNN_model():
    seq = FNN.model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    return seq

def CNN_model():
    seq = CNN.model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    return seq

def LSTM_model():
    seq = LSTM.model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    return seq

# monthly sst parameters setting
epochs = 1000
batch_size = 100
validation_split = 0.1
train_length = 1800
len_seq = 1980
len_frame = 9
start_seq = 1801
end_seq = 1968
# begin_pred_seq = 6
# end_pred_seq = 12

fold_name = "model_"+str(epochs)+"_epochs"
# DATA_PATH = '../../../../dataset/sst_grid_1/sst_monthly_185001_201512.npy'
DATA_PATH = 'monthly_sst+1.npy'

def main():
    os.makedirs(fold_name)
    # fit model
    file_path = fold_name +'/'+fold_name +".h5"
    log_file_path = fold_name+'/'+fold_name +".log"
    log = open(log_file_path,'w')

    # model setting
    seq = CovLSTM2D_model()
    with redirect_stdout(log):
        seq.summary()

    # seq = LSTM_model()
    # with redirect_stdout(log):
    #     seq.summary()

    # TODO
    # seq = STResNet_model()

    # sst_grid, train_X, train_Y= pp.load_data_convlstm_monthly(train_length) # From .mat file
    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file

    # normalization, data for ConvLSTM Model -9 ahead
    train_X = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    train_Y = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    sst_grid = np.zeros((len_seq+9, len_frame, 10, 50, 1), dtype=np.float)
    for i in range(len_seq):
        for k in range(len_frame):
            train_X[i,k,::,::,0] = pp.normalization(train_X_raw[i,k,::,::,0])
            train_Y[i,k,::,::,0] = pp.normalization(train_Y_raw[i,k,::,::,0])
            sst_grid[i,k,::,::,0] = pp.normalization(sst_grid_raw[i,k,::,::,0])

    for m in range(len_frame):
        sst_grid[len_seq,m,::,::,0] = pp.normalization(sst_grid_raw[len_seq,m,::,::,0])

    # # normalization, data for CNN Model
    # train_X = np.zeros((len_seq, len_frame, 10, 50), dtype=np.float)
    # train_Y = np.zeros((len_seq, len_frame, 10, 50), dtype=np.float)
    # sst_grid = np.zeros((len_seq+12, len_frame, 10, 50), dtype=np.float)
    # for i in range(len_seq):
    #     for k in range(len_frame):
    #         train_X[i,k,::,::] = pp.normalization(train_X_raw[i,k,::,::,0])
    #         train_Y[i,k,::,::] = pp.normalization(train_Y_raw[i,k,::,::,0])
    #         sst_grid[i,k,::,::] = pp.normalization(sst_grid_raw[i,k,::,::,0])
    # for m in range(len_frame):
    #     sst_grid[len_seq,m,::,::] = pp.normalization(sst_grid_raw[len_seq,m,::,::,0])


    seq = multi_gpu_model(seq, gpus=2)
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    seq.compile(loss='mse', optimizer='adam')

    if not os.path.exists(file_path):

        # ConvLSTM Model
        history = seq.fit(train_X[:train_length], train_Y[:train_length],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split)

        # # FC-LSTM Model
        # print(train_X[:train_length].shape)
        # history = seq.fit(train_X[:train_length], train_Y[:train_length],
        #             batch_size=batch_size,
        #             epochs=epochs)
        #
        seq.save(file_path)
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper left')
        pyplot.savefig(fold_name + '/%i_epoch_loss.png' % epochs)
    else:
        seq = load_model(file_path)

    model_sum_rmse_12 = 0
    base_sum_rmse_12 = 0
    model_sum_mae_12 = 0
    base_sum_mae_12 = 0
    model_sum_mape_12 = 0
    base_sum_mape_12 = 0

    for k in range(start_seq, end_seq):

        # # direct multi-step forecasting
        # pred_sequence_raw = sst_grid[k][::, ::, ::, ::]
        # new_frame = seq.predict(pred_sequence_raw[np.newaxis, ::, ::, ::, ::])
        # pred_sequence = new_frame[0]
        # act_sequence = sst_grid[k+12][::, ::, ::, ::]
        # for i in range(12):
        #     baseline_frame = pp.inverse_normalization(pred_sequence_raw[i, ::, ::, 0])
        #     pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
        #     act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
        #
        #     model_rmse = mean_squared_error(act_toplot, pred_toplot)
        #     baseline_rmse = mean_squared_error(act_toplot, baseline_frame)
        #
        #     model_mae = mean_absolute_error(act_toplot, pred_toplot)
        #     baseline_mae = mean_absolute_error(act_toplot, baseline_frame)
        #
        #     model_mape = pp.mean_absolute_percentage_error(act_toplot, pred_toplot)
        #     baseline_mape = pp.mean_absolute_percentage_error(act_toplot, baseline_frame)
        #
        #     model_sum_rmse_12, base_sum_rmse_12 = model_sum_rmse_12 + model_rmse, base_sum_rmse_12 + baseline_rmse
        #     model_sum_mae_12, base_sum_mae_12 = model_sum_mae_12 + model_mae, base_sum_mae_12 + baseline_mae
        #     model_sum_mape_12, base_sum_mape_12 = model_sum_mape_12 + model_mape, base_sum_mape_12 + baseline_mape

        # rolling-forecasting with -6 steps
        pred_sequence_raw = sst_grid[k][::, ::, ::, ::]
        pred_sequence = sst_grid[k][::, ::, ::, ::]
        act_sequence = sst_grid[k+9][::, ::, ::, ::]
        for j in range(9):
            new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
            # TODO why?? ref: keras conv_lstm.py demo
            new = new_frame[::, -1, ::, ::, ::]
            pred_sequence = np.concatenate((pred_sequence, new), axis=0)

            baseline_frame = pp.inverse_normalization(pred_sequence_raw[j, ::, ::, 0])
            pred_toplot = pp.inverse_normalization(pred_sequence[-1, ::, ::, 0])
            act_toplot = pp.inverse_normalization(act_sequence[j, ::, ::, 0])
            pred_sequence = pred_sequence[1:10, ::, ::, ::]

            model_rmse = mean_squared_error(act_toplot, pred_toplot)
            baseline_rmse = mean_squared_error(act_toplot, baseline_frame)

            model_mae = mean_absolute_error(act_toplot, pred_toplot)
            baseline_mae = mean_absolute_error(act_toplot, baseline_frame)

            model_mape = pp.mean_absolute_percentage_error(act_toplot, pred_toplot)
            baseline_mape = pp.mean_absolute_percentage_error(act_toplot, baseline_frame)

            model_sum_rmse_12, base_sum_rmse_12 = model_sum_rmse_12 + model_rmse, base_sum_rmse_12 + baseline_rmse
            model_sum_mae_12, base_sum_mae_12 = model_sum_mae_12 + model_mae, base_sum_mae_12 + baseline_mae
            model_sum_mape_12, base_sum_mape_12 = model_sum_mape_12 + model_mape, base_sum_mape_12 + baseline_mape

    print("="*10)
    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse_12/(9*(end_seq-start_seq)))))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse_12/(9*(end_seq-start_seq)))))
    print("Total Model MAE: %s" % (model_sum_mae_12/(9*(end_seq-start_seq))))
    print("Total Baseline MAE: %s" % (base_sum_mae_12/(9*(end_seq-start_seq))))
    print("Model MAPE: %s" % (model_sum_mape_12/(9*(end_seq-start_seq))))
    print("Baseline MAPE: %s" % (base_sum_mape_12/(9*(end_seq-start_seq))))

    log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse_12/(9*(end_seq-start_seq)))))
    log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse_12/(9*(end_seq-start_seq)))))
    log.write("\nTotal Model MAE: %s" % (model_sum_mae_12/(9*(end_seq-start_seq))))
    log.write("\nTotal Baseline MAE: %s" % (base_sum_mae_12/(9*(end_seq-start_seq))))
    log.write("\nModel MAPE: %s" % (model_sum_mape_12/(9*(end_seq-start_seq))))
    log.write("\nBaseline MAPE: %s" % (base_sum_mape_12/(9*(end_seq-start_seq))))

    log.close()

    # # visulize one seq (Rolling -forecast)
    # pred_sequence = sst_grid[which_seq][:12, ::, ::, ::]
    # for j in range(12):
    #     new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
    #     # TODO why?? ref: keras conv_lstm.py demo
    #     new = new_frame[::, -1, ::, ::, ::]
    #     pred_sequence = np.concatenate((pred_sequence, new), axis=0)

    for k in range(end_seq-10, end_seq):
        pred_sequence_raw = sst_grid[k][::, ::, ::, ::]
        new_frame = seq.predict(pred_sequence_raw[np.newaxis, ::, ::, ::, ::])
        pred_sequence = new_frame[0]
        act_sequence = sst_grid[k+9][::, ::, ::, ::]
        for i in range(9):
        # for i in range(begin_pred_seq, end_pred_seq):
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
            plt.savefig(fold_name + '/%i_%i_animate.png' % ((k + 1), (i + 1)))

if __name__ == '__main__':
    main()
