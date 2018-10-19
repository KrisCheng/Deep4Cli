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
epochs = 20000
batch_size = 10
validation_split = 0.05
train_length = 160
len_year = 167
start_year = 160
begin_pred_month = 6
end_pred_month = 12
# which_year = 161 # year to visulization

# # daily ssta parameters setting
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

def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = ylen
    while(True):
        i = randint(0, loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

def main():

    # os.makedirs(fold_name)
    # # fit model
    file_path = fold_name+'/'+fold_name +".h5"
    file_path = "20000epoch.h5"
    #
    # log_file_path = fold_name+'/'+fold_name +".log"
    # log = open(log_file_path,'w')

    # model setting
    seq = CovLSTM2D_model()
    # with redirect_stdout(log):
    #     seq.summary()
    # TODO
    # seq = STResNet_model()

    sst_grid, train_X, train_Y= pp.load_data_convlstm_monthly(train_length)
    # sst_grid, train_X, train_Y= pp.load_data_convlstm_daily(train_length)

    seq = multi_gpu_model(seq, gpus=2)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    seq.compile(loss='mse', optimizer='sgd')

    if not os.path.exists(file_path):
        # ConvLSTM Model
        history = seq.fit_generator(generate_batch_data_random(train_X, train_Y, batch_size),
                    samples_per_epoch = len(train_Y),
                    epochs=epochs)
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

    for k in range(start_year, len_year):
        pred_sequence = sst_grid[k][:6, ::, ::, ::]
        for j in range(6):
            new_frame = seq.predict(pred_sequence[np.newaxis, ::, ::, ::, ::])
            # TODO why?? ref: keras conv_lstm.py demo
            new = new_frame[::, -1, ::, ::, ::]
            pred_sequence = np.concatenate((pred_sequence, new), axis=0)

        # And then compare the predictions with the ground truth
        act_sequence = sst_grid[k][::, ::, ::, ::]

        for i in range(begin_pred_month, end_pred_month):
            fig = plt.figure(figsize=(16, 8))

            ax = fig.add_subplot(321)
            ax.text(1, 3, 'Prediction', fontsize=12)
            pred_toplot = pp.inverse_normalization(pred_sequence[i, ::, ::, 0])
            plt.imshow(pred_toplot)
            cbar = plt.colorbar(plt.imshow(pred_toplot), orientation='horizontal')
            cbar.set_label('°C',fontsize=12)

            # 将预测当年6月份数据作为baseline
            baseline_frame = pp.inverse_normalization(act_sequence[5, ::, ::, 0])
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

            model_rmse = sqrt(mean_squared_error(act_toplot, pred_toplot))
            baseline_rmse = sqrt(mean_squared_error(act_toplot, baseline_frame))

            model_sum_loss, base_sum_loss = model_sum_loss + model_rmse, base_sum_loss + baseline_rmse
            plt.savefig(fold_name + '/%i_%i_animate.png' % ((k + 1), (i + 1)))

    print("="*10)
    print("Total Model RMSE: %s" % (model_sum_loss/(6*(len_year-start_year))))
    print("Total Baseline RMSE: %s" % (base_sum_loss/(6*(len_year-start_year))))
    log.write("\nTotal Model RMSE: %s" % (model_sum_loss/(6*(len_year-start_year))))
    log.write("\nTotal Baseline RMSE: %s" % (base_sum_loss/(6*(len_year-start_year))))

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
    #
    # log.close()

if __name__ == '__main__':
    main()
