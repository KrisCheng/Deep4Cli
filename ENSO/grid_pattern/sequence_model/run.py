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
import os
import os.path
from matplotlib import pyplot
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras import optimizers
from sklearn.metrics import mean_squared_error
from math import sqrt
from contextlib import redirect_stdout
from keras import backend as K
from random import randint

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

# monthly sst parameters setting
epochs = 5000
batch_size = 50
validation_split = 0.05
train_length = 1800
len_seq = 1980
start_seq = 1801
begin_pred_seq = 12
end_pred_seq = 24
# which_year = 161 # year to visulization

# # daily ssta parameters setting TODO
# epochs = 2
# batch_size = 1
# validation_split = 0.5
# train_length = 4
# len_year = 36
# start_year = 32
# begin_pred_month = 320
# end_pred_month = 365
# # which_year = 161 # year to visulization

fold_name = "model_"+str(epochs)+"_epochs"
DATA_PATH = '../../../../dataset/sst_grid_1/sst_monthly_185001_201512.npy'
def main():

    os.makedirs(fold_name)
    # fit model
    file_path = fold_name+'/'+fold_name +".h5"

    log_file_path = fold_name+'/'+fold_name +".log"
    log = open(log_file_path,'w')

    # model setting
    seq = CovLSTM2D_model()
    with redirect_stdout(log):
        seq.summary()
    # TODO
    # seq = STResNet_model()

    # sst_grid, train_X, train_Y= pp.load_data_convlstm_monthly(train_length) # From .mat file
    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file
    # sst_grid, train_X, train_Y= pp.load_data_convlstm_daily(train_length) TODO

    # normalization
    train_X = np.zeros((1980, 24, 10, 50, 1), dtype=np.float)
    train_Y = np.zeros((1980, 24, 10, 50, 1), dtype=np.float)
    sst_grid = np.zeros((1981, 24, 10, 50, 1), dtype=np.float)
    for i in range(1980):
        for k in range(24):
            train_X[i,k,::,::,0] = pp.normalization(train_X_raw[i,k,::,::,0])
            train_Y[i,k,::,::,0] = pp.normalization(train_Y_raw[i,k,::,::,0])
            sst_grid[i,k,::,::,0] = pp.normalization(sst_grid_raw[i,k,::,::,0])
    for m in range(24):
        sst_grid[1980,m,::,::,0] = pp.normalization(sst_grid_raw[1980,m,::,::,0])

    seq = multi_gpu_model(seq, gpus=2)
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    seq.compile(loss='mse', optimizer='adadelta')

    if not os.path.exists(file_path):
        # ConvLSTM Model
        history = seq.fit(train_X[:train_length], train_Y[:train_length],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.05)
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

    # Evaluation part (SST, monthly)
    # Testing the network on new monthly SST distribution
    # feed it with the first 6 patterns
    # predict the next 6 pattern

    model_sum_loss = 0
    base_sum_loss = 0

    # for k in range(start_seq, len_seq):
    pred_sequence = sst_grid[1900][:12, ::, ::, ::]
    for j in range(12):
        new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
        # TODO why?? ref: keras conv_lstm.py demo
        new = new_frame[::, -1, ::, ::, ::]
        pred_sequence = np.concatenate((pred_sequence, new), axis=0)

    for k in range(start_seq, len_seq):
        act_sequence = sst_grid[k][::, ::, ::, ::]
        pred_sequence = sst_grid[k][:12, ::, ::, ::]
        for j in range(12):
            new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
            # TODO why?? ref: keras conv_lstm.py demo
            new = new_frame[::, -1, ::, ::, ::]
            pred_sequence = np.concatenate((pred_sequence, new), axis=0)
        for i in range(begin_pred_seq, end_pred_seq):
            baseline_frame = pp.inverse_normalization(act_sequence[11, ::, ::, 0])
            pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
            act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
            model_rmse = sqrt(mean_squared_error(act_toplot, pred_toplot))
            baseline_rmse = sqrt(mean_squared_error(act_toplot, baseline_frame))
            model_sum_loss, base_sum_loss = model_sum_loss + model_rmse, base_sum_loss + baseline_rmse

    print("="*10)
    print("Total Model RMSE: %s" % (model_sum_loss/(12*(len_seq-start_seq))))
    print("Total Baseline RMSE: %s" % (base_sum_loss/(12*(len_seq-start_seq))))
    log.write("\nTotal Model RMSE: %s" % (model_sum_loss/(12*(len_seq-start_seq))))
    log.write("\nTotal Baseline RMSE: %s" % (base_sum_loss/(12*(len_seq-start_seq))))

    # And then compare the predictions with the ground truth
    act_sequence = sst_grid[1900][::, ::, ::, ::]
    for i in range(begin_pred_seq, end_pred_seq):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(321)
        ax.text(1, 3, 'Prediction', fontsize=12)
        pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
        plt.imshow(pred_toplot)
        cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
        cbar.set_label('°C',fontsize=12)
        # 将预测seq-12数据作为baseline
        baseline_frame = pp.inverse_normalization(act_sequence[11, ::, ::, 0])
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
        plt.savefig(fold_name + '/%i_animate.png' % ( (i + 1)))

    # Evaluation part (SSTA ,daily)
    # Testing the network on new monthly SSTA distribution
    # feed it with the first 320 patterns
    # predict the next 45 pattern

    # for k in range(start_year, len_year):
    #     pred_sequence = sst_grid[k][:6, ::, ::, ::]
    #     for j in range(6):
    #         new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
    #         # TODO why?? ref: keras conv_lstm.py demo
    #         new = new_frame[::, -1, ::, ::, ::]
    #         pred_sequence = np.concatenate((pred_sequence, new), axis=0)
    #
    #     # And then compare the predictions with the ground truth
    #     act_sequence = sst_grid[k][::, ::, ::, ::]
    #
    #     for i in range(begin_pred_month, end_pred_month):
    #         fig = plt.figure(figsize=(16, 8))
    #
    #         ax = fig.add_subplot(321)
    #         ax.text(1, 3, 'Prediction', fontsize=12)
    #         pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
    #         plt.imshow(pred_toplot)
    #         cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
    #         cbar.set_label('°C',fontsize=12)
    #
    #         # 将预测当年6月份数据作为baseline
    #         baseline_frame = pp.inverse_normalization(act_sequence[5, ::, ::, 0])
    #         ax = fig.add_subplot(322)
    #         plt.text(1, 3, 'Baseline', fontsize=12)
    #         plt.imshow(baseline_frame)
    #         cbar = plt.colorbar(plt.imshow(baseline_frame), orientation='horizontal')
    #         cbar.set_label('°C',fontsize=12)
    #
    #         ax = fig.add_subplot(323)
    #         plt.text(1, 3, 'Ground truth', fontsize=12)
    #         act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
    #         plt.imshow(act_toplot)
    #         cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
    #         cbar.set_label('°C',fontsize=12)
    #
    #         ax = fig.add_subplot(324)
    #         plt.text(1, 3, 'Ground truth', fontsize=12)
    #         act_toplot = pp.inverse_normalization(act_sequence[i, ::, ::, 0])
    #         plt.imshow(act_toplot)
    #         cbar = plt.colorbar(plt.imshow(act_toplot), orientation='horizontal')
    #         cbar.set_label('°C',fontsize=12)
    #
    #         ax = fig.add_subplot(325)
    #         plt.text(1, 3, 'Diff_Pred', fontsize=12)
    #         diff_toplot = act_toplot - pred_toplot
    #         plt.imshow(diff_toplot)
    #         cbar = plt.colorbar(plt.imshow(diff_toplot), orientation='horizontal')
    #         cbar.set_label('°C',fontsize=12)
    #
    #         ax = fig.add_subplot(326)
    #         plt.text(1, 3, 'Diff_Base', fontsize=12)
    #         diff_toplot = act_toplot - baseline_frame
    #         plt.imshow(diff_toplot)
    #         cbar = plt.colorbar(plt.imshow(diff_toplot), orientation='horizontal')
    #         cbar.set_label('°C',fontsize=12)
    #
    #         model_rmse = sqrt(mean_squared_error(act_toplot, pred_toplot))
    #         baseline_rmse = sqrt(mean_squared_error(act_toplot, baseline_frame))
    #
    #         model_sum_loss, base_sum_loss = model_sum_loss + model_rmse, base_sum_loss + baseline_rmse
    #         plt.savefig(fold_name + '/%i_%i_animate.png' % ((k + 1), (i + 1)))
    #
    # print("="*10)
    # print("Total Model RMSE: %s" % (model_sum_loss/(6*(len_year-start_year))))
    # print("Total Baseline RMSE: %s" % (base_sum_loss/(6*(len_year-start_year))))
    # log.write("\nTotal Model RMSE: %s" % (model_sum_loss/(6*(len_year-start_year))))
    # log.write("\nTotal Baseline RMSE: %s" % (base_sum_loss/(6*(len_year-start_year))))

    log.close()

if __name__ == '__main__':
    main()
