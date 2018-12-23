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
epochs = 201
batch_size = 100*500
validation_split = 0.1
train_length = 1800
len_seq = 1980
len_frame = 12
start_seq = 1801
end_seq = 1968
height, width = 10, 50
point_x, point_y = 2, 2

DATA_PATH = 'monthly_sst+1.npy'
fold_name = "model_"+str(epochs)+"_epochs_"+str(len_frame)

def model():
    # Model
    seq = Sequential()
    seq.add(LSTM(units = 20, input_shape=(len_frame, 1), activation='relu', return_sequences=True))
    seq.add(LSTM(20, activation='relu'))
    seq.add(Dense(1))
    return seq

def main():
    # os.makedirs(fold_name)
    # fit model
    file_path = fold_name +'/'+fold_name +".h5"
    log_file_path = fold_name+'/'+fold_name +".log"
    log = open(log_file_path,'w')

    train_X_raw, train_Y_raw, sst_grid = np.load(DATA_PATH)  # from .npy file

    # LSTM Model -n ahead
    train_X = np.zeros((train_length*height*width, len_frame, 1), dtype=np.float)
    train_Y = np.zeros((train_length*height*width, 1), dtype=np.float)

    for i in range(train_length):
        for m in range(width):
            for n in range(height):
                train_X[i+m+n,::,0] = sst_grid[i,0:len_frame,n,m,0]
                train_Y[i+m+n,0] = sst_grid[i+1,len_frame-1,n,m,0]

    model_sum_rmse = 0
    base_sum_rmse = 0
    model_sum_mae = 0
    base_sum_mae = 0
    model_sum_mape = 0
    base_sum_mape = 0

    single_point_model_sum_rmse = 0
    single_point_base_sum_rmse = 0

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
            # single-point
            history = np.array([float(x) for x in pred_sequence[::,point_x,point_y,::]])
            history = history.reshape(1, len_frame, 1)
            predict = seq.predict(history, verbose=0)
            pred_toplot_single = predict[0]
            baseline_frame_single = pred_sequence_raw[j, point_x, point_y, 0]
            act_toplot_single = act_sequence[j, point_x, point_y, 0]

            single_model_rmse = (act_toplot_single-pred_toplot_single)**2
            single_base_rmse = (act_toplot_single-baseline_frame_single)**2
            single_point_model_sum_rmse = single_point_model_sum_rmse + single_model_rmse
            single_point_base_sum_rmse = single_point_base_sum_rmse + single_base_rmse

    print("="*10)
    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Model MAE: %s" % (model_sum_mae/(len_frame*(end_seq-start_seq))))
    print("Total Baseline MAE: %s" % (base_sum_mae/(len_frame*(end_seq-start_seq))))
    print("Model MAPE: %s" % (model_sum_mape/(len_frame*(end_seq-start_seq))))
    print("Baseline MAPE: %s" % (base_sum_mape/(len_frame*(end_seq-start_seq))))
    print("Single Model RMSE: %s" % (sqrt(single_point_model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Single Baseline RMSE: %s" % (sqrt(single_point_base_sum_rmse/(len_frame*(end_seq-start_seq)))))

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
