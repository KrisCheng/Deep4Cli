'''
Desc: LSTM Model, single point .
Author: Kris Peng
'''

from keras.models import Sequential
from keras.layers.convolutional import Conv3D, Conv2D
from keras.layers import Dense
from keras.layers import LSTM, Flatten
from keras.models import load_model
from keras.utils import multi_gpu_model
import preprocessing as pp
import pylab as plt
import pandas as pd
import numpy as np
import sys
import os
import os.path
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from contextlib import redirect_stdout
sys.setrecursionlimit(100000000)

# monthly sst parameters setting
epochs = 1000
batch_size = 100*500
validation_split = 0.1
train_length = 1800
len_seq = 1980
len_frame = 6
start_seq = 1801
end_seq = 1968
height, width = 10, 50

DATA_PATH = 'monthly_sst+1.npy'
fold_name = "model_"+str(epochs)+"_epochs_"+str(len_frame)


def model():
    # Model
    seq = Sequential()
    seq.add(LSTM(units = 300, input_shape=(len_frame,1), activation='relu', return_sequences=True))
    seq.add(LSTM(300, return_sequences=True))
    seq.add(Dense(1))
    return seq

def main():
    os.makedirs(fold_name)
    # fit model
    file_path = fold_name +'/'+fold_name +".h5"
    log_file_path = fold_name+'/'+fold_name +".log"
    log = open(log_file_path,'w')

    train_X_raw, train_Y_raw, sst_grid = np.load(DATA_PATH)  # from .npy file

    # LSTM Model -n ahead
    train_X = np.zeros((train_length*height*width, len_frame, 1), dtype=np.float)
    train_Y = np.zeros((train_length*height*width, len_frame, 1), dtype=np.float)

    for i in range(train_length):
        for m in range(width):
            for n in range(height):
                train_X[i+m+n,::,0] = sst_grid[i,0:len_frame,n,m,0]
                train_Y[i+m+n,::,0] = sst_grid[i+1,0:len_frame,n,m,0]

    model_sum_rmse = 0
    base_sum_rmse = 0
    model_sum_mae = 0
    base_sum_mae = 0
    model_sum_mape = 0
    base_sum_mape = 0

    seq = model()
    print('=' * 10)
    print(seq.summary())
    print('=' * 10)
    with redirect_stdout(log):
        seq.summary()

    seq = multi_gpu_model(seq, gpus=2)
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    seq.compile(loss='mse', optimizer='adam')

    if not os.path.exists(file_path):
        # LSTM Model
        history = seq.fit(train_X, train_Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split)
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

    for k in range(start_seq, end_seq):
        print(k)
        # rolling-forecasting with -N steps
        pred_sequence_raw = sst_grid[k][0:len_frame, ::, ::, ::]
        pred_sequence = sst_grid[k][0:len_frame, ::, ::, ::]
        act_sequence = sst_grid[k+len_frame][0:len_frame, ::, ::, ::]
        for j in range(len_frame):
            pred_toplot = np.zeros((1,10,50,1), dtype=np.float)
            # single point prediction
            for i in range(width):
                for m in range(height):
                    history = np.array([float(x) for x in pred_sequence[::,m,i,::]])
                    history = history.reshape(1,len_frame,1)
                    predict = seq.predict(history, verbose=0)
                    pred_toplot[0][m][i][0] = predict[0,0,-1]
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

            model_mape = pp.mean_absolute_percentage_error(act_toplot, pred_toplot)
            baseline_mape = pp.mean_absolute_percentage_error(act_toplot, baseline_frame)

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
        plt.savefig(fold_name + '/%i_%i_animate.png' % ((k + 1), (j + 1)))

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
