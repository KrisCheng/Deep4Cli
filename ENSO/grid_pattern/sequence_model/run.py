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
import metric
import ConvLSTM2D
import STResNet, FNN, CNN
import os.path
import sys
from matplotlib import pyplot
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from contextlib import redirect_stdout
sys.setrecursionlimit(100000000)

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

# monthly sst parameters setting
epochs = 300
batch_size = 100
validation_split = 0.1
train_length = 1800
len_seq = 1980
len_frame = 12
start_seq = 1801
end_seq = 1968
point_x, point_y = 2, 2

fold_name = "model_"+str(epochs)+"_epochs_"+str(len_frame)
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

    # TODO
    # seq = STResNet_model()

    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file

    # normalization, data for ConvLSTM Model -n ahead -5 dimension
    train_X = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    train_Y = np.zeros((len_seq, len_frame, 10, 50, 1), dtype=np.float)
    sst_grid = np.zeros((len_seq+len_frame, len_frame, 10, 50, 1), dtype=np.float)
    for i in range(len_seq):
        for k in range(len_frame):
            train_X[i,k,::,::,0] = pp.normalization(train_X_raw[i,k,::,::,0])
            train_Y[i,k,::,::,0] = pp.normalization(train_Y_raw[i,k,::,::,0])
            sst_grid[i,k,::,::,0] = pp.normalization(sst_grid_raw[i,k,::,::,0])

    # # normalization, data for ConvLSTM Model -n ahead -4 dimension
    # train_X = np.zeros((len_seq, len_frame, 10, 50), dtype=np.float)
    # train_Y = np.zeros((len_seq, len_frame, 10, 50), dtype=np.float)
    # sst_grid = np.zeros((len_seq+len_frame, len_frame, 10, 50), dtype=np.float)
    # for i in range(len_seq):
    #     for k in range(len_frame):
    #         train_X[i,k,::,::] = pp.normalization(train_X_raw[i,k,::,::])
    #         train_Y[i,k,::,::] = pp.normalization(train_Y_raw[i,k,::,::])
    #         sst_grid[i,k,::,::] = pp.normalization(sst_grid_raw[i,k,::,::])
    # for m in range(len_frame):
    #     sst_grid[len_seq,m,::,::] = pp.normalization(sst_grid_raw[len_seq,m,::,::])

    seq = multi_gpu_model(seq, gpus=2)
    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # rmsprop = optimizers.RMSprop(lr=0.1)
    seq.compile(loss="mse", optimizer='adam')

    if not os.path.exists(file_path):
        # ConvLSTM Model
        history = seq.fit(train_X[:train_length], train_Y[:train_length],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split)
        seq.save(file_path)
        pyplot.plot(history.history['loss'])
        log.write("\n train_loss=========")
        log.write("\n %s" % history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        log.write("\n\n\n val_loss=========")
        log.write("\n %s" % history.history['val_loss'])
        pyplot.title('model loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper left')
        pyplot.savefig(fold_name + '/%i_epoch_loss.png' % epochs)
    else:
        seq = load_model(file_path)

    model_sum_rmse = 0
    base_sum_rmse = 0
    model_sum_mae = 0
    base_sum_mae = 0
    model_sum_mape = 0
    base_sum_mape = 0

    single_point_model_sum_rmse = 0
    single_point_base_sum_rmse = 0

    for k in range(start_seq, end_seq):
        # rolling-forecasting with -n steps
        model_sum_rmse_current = 0
        base_sum_rmse_current = 0
        model_sum_mae_current = 0
        base_sum_mae_current = 0
        model_sum_mape_current = 0
        base_sum_mape_current = 0

        pred_sequence_raw = sst_grid[k][::, ::, ::, ::]
        pred_sequence = sst_grid[k][::, ::, ::, ::]
        act_sequence = sst_grid[k+len_frame][::, ::, ::, ::]

        for j in range(len_frame):
            new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
            # TODO why?? ref: keras conv_lstm.py demo
            new = new_frame[::, -1, ::, ::, ::]
            pred_sequence = np.concatenate((pred_sequence, new), axis=0)

            baseline_frame = pp.inverse_normalization(pred_sequence_raw[j, ::, ::, 0])
            pred_toplot = pp.inverse_normalization(pred_sequence[-1, ::, ::, 0])
            act_toplot = pp.inverse_normalization(act_sequence[j, ::, ::, 0])

            pred_sequence = pred_sequence[1:len_frame+1, ::, ::, ::]

            model_rmse = mean_squared_error(act_toplot, pred_toplot)
            # print(model_rmse)
            baseline_rmse = mean_squared_error(act_toplot, baseline_frame)

            model_mae = mean_absolute_error(act_toplot, pred_toplot)
            baseline_mae = mean_absolute_error(act_toplot, baseline_frame)

            model_mape = pp.mean_absolute_percentage_error(act_toplot, pred_toplot)
            baseline_mape = pp.mean_absolute_percentage_error(act_toplot, baseline_frame)

            model_sum_rmse, base_sum_rmse = model_sum_rmse + model_rmse, base_sum_rmse + baseline_rmse
            model_sum_mae, base_sum_mae = model_sum_mae + model_mae, base_sum_mae + baseline_mae
            model_sum_mape, base_sum_mape = model_sum_mape + model_mape, base_sum_mape + baseline_mape

            model_sum_rmse_current, base_sum_rmse_current = model_sum_rmse_current + model_rmse, base_sum_rmse_current + baseline_rmse
            model_sum_mae_current, base_sum_mae_current = model_sum_mae_current + model_mae, base_sum_mae_current + baseline_mae
            model_sum_mape_current, base_sum_mape_current = model_sum_mape_current + model_mape, base_sum_mape_current + baseline_mape

            single_model_rmse = (act_toplot[point_x, point_y]-pred_toplot[point_x, point_y])**2
            single_base_rmse = (act_toplot[point_x, point_y]-baseline_frame[point_x, point_y])**2

            single_point_model_sum_rmse = single_point_model_sum_rmse + single_model_rmse
            single_point_base_sum_rmse = single_point_base_sum_rmse + single_base_rmse

        log.write("\n\n ============")
        log.write("\n Round: %s" % str(k+1))
        log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse_current/len_frame)))
        log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse_current/len_frame)))
        log.write("\nTotal Model MAE: %s" % (model_sum_mae_current/len_frame))
        log.write("\nTotal Baseline MAE: %s" % (base_sum_mae_current/len_frame))
        log.write("\nModel MAPE: %s" % (model_sum_mape_current/len_frame))
        log.write("\nBaseline MAPE: %s" % (base_sum_mape_current/len_frame))

        print("============")
        print("Round: %s" % str(k+1))
        print("Total Model RMSE: %s" % (sqrt(model_sum_rmse_current/len_frame)))
        print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse_current/len_frame)))
        print("Total Model MAE: %s" % (model_sum_mae_current/len_frame))
        print("Total Baseline MAE: %s" % (base_sum_mae_current/len_frame))
        print("Model MAPE: %s" % (model_sum_mape_current/len_frame))
        print("Baseline MAPE: %s" % (base_sum_mape_current/len_frame))

    print("="*10)
    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Model MAE: %s" % (model_sum_mae/(len_frame*(end_seq-start_seq))))
    print("Total Baseline MAE: %s" % (base_sum_mae/(len_frame*(end_seq-start_seq))))
    print("Model MAPE: %s" % (model_sum_mape/(len_frame*(end_seq-start_seq))))
    print("Baseline MAPE: %s" % (base_sum_mape/(len_frame*(end_seq-start_seq))))

    print("Single Model RMSE: %s" % (sqrt(single_point_model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Single Baseline RMSE: %s" % (sqrt(single_point_base_sum_rmse/(len_frame*(end_seq-start_seq)))))

    log.write("\n\n Total:")
    log.write("\nTotal Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    log.write("\nTotal Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(end_seq-start_seq)))))
    log.write("\nTotal Model MAE: %s" % (model_sum_mae/(len_frame*(end_seq-start_seq))))
    log.write("\nTotal Baseline MAE: %s" % (base_sum_mae/(len_frame*(end_seq-start_seq))))
    log.write("\nModel MAPE: %s" % (model_sum_mape/(len_frame*(end_seq-start_seq))))
    log.write("\nBaseline MAPE: %s" % (single_point_base_sum_rmse/(len_frame*(end_seq-start_seq))))
    log.close()

    # # visulize one seq (Rolling -forecast)
    # pred_sequence = sst_grid[which_seq][:12, ::, ::, ::]
    # for j in range(12):
    #     new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
    #     # TODO why?? ref: keras conv_lstm.py demo
    #     new = new_frame[::, -1, ::, ::, ::]
    #     pred_sequence = np.concatenate((pred_sequence, new), axis=0)

    for k in range(start_seq, end_seq, 80):
        pred_sequence_raw = sst_grid[k][::, ::, ::, ::]
        new_frame = seq.predict(pred_sequence_raw[np.newaxis, ::, ::, ::, ::])
        pred_sequence = new_frame[0]
        act_sequence = sst_grid[k+len_frame][::, ::, ::, ::]
        for i in range(len_frame):
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
            plt.savefig(fold_name + '/%s_%s_animate.png' % (str(k + 1), str(i + 1)))

if __name__ == '__main__':
    main()
