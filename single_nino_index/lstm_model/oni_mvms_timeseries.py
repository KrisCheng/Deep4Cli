#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  apply the LSTM model, on different nino(1+2, 3, 4, 3.4) index dataset, based on keras (from 1870~2017, monthly).
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Reference: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
           https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
           https://github.com/Yongyao/enso-forcasting/blob/master/inotebook/LSTM%20modeling.ipynb
Author: Kris Peng
Date: 2018/05/18
'''

from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from matplotlib import pyplot
import os.path

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i).iloc[:,-1])
        if i == 0:
            names += ['VAR(t)']
        else:
            names += ['VAR(t+%d)' % i]

    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def fit_lstm(train, n_lag, n_ahead, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, :-n_ahead], train[:, -n_ahead:]
    # X = X.reshape(X.shape[0], 1, X.shape[1])
    X = X.reshape(X.shape[0], n_lag, int(X.shape[1]/n_lag))
    # y = y.reshape(y.shape[0], 1, n_ahead)

    # design network
    model = Sequential()
    # single layer
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    # multi layer
    # model.add(LSTM(n_neurons, return_sequences=True, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    # model.add(LSTM(n_neurons))
    
    model.add(Dense(n_ahead))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, X, n_batch, n_lag):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, n_lag, int(len(X)/n_lag))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

def make_forecasts(model, n_batch, train, test, n_lag, n_ahead):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, :-n_ahead]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch, n_lag)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(y, forecasts, n_lag, n_seq):
    sum_rmse = []
    for i in range(n_seq):
        actual = [row[i] for row in y]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('%.3f' % ((i+1), rmse))
        sum_rmse.append(rmse) 
    print("%.3f" % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5])/6))
    print("%.3f" % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5]+sum_rmse[6]+sum_rmse[7]+sum_rmse[8])/9))
    print("%.3f" % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5]+sum_rmse[6]+sum_rmse[7]+sum_rmse[8]+sum_rmse[9]+sum_rmse[10]+sum_rmse[11])/12))
        
# plot the forecasts in the context of the original dataset, multiple segments
def plot_forecasts(series, forecasts, n_test, linestyle = None):
    # plot the entire dataset in blue
    pyplot.figure()
    if linestyle==None:
        pyplot.plot(series, label='observed')
    else:
        pyplot.plot(series, linestyle, label='observed')
    pyplot.title("oni_multivariate_multistep_timeseries")
    pyplot.legend(loc='upper right')
    # plot the forecasts in red
    for i in range(len(forecasts)):
        if i%n_seq ==0 and i != 0: # this ensures not all segements are plotted, instead it is plotted every n_ahead
            off_s = len(series) - n_test + i - 1
            off_e = off_s + len(forecasts[i]) + 1
            xaxis = [x for x in range(off_s, off_e)]
            yaxis = [series[off_s]] + forecasts[i] 
            pyplot.plot(xaxis, yaxis, 'r')
            # print(off_s, off_e)
    # show the plot
    pyplot.show()

# load dataset
df = read_csv('../../data/oni/csv/all_nino_anomaly.csv', header=0, index_col=0)
df = (df - df.mean()) / df.std()

cols = df.columns.tolist()
cols = cols[1:] + cols[:1]

df = df[cols]

enso = df.values.astype('float32')

# print(enso)

# parameter setting
n_lag = 12
n_seq = 12
n_test = 96
n_epochs = 10
n_neurons = 1
n_batch = 1

reframed = series_to_supervised(enso, n_lag, n_seq)
# print(reframed)
# print(reframed.head())

# Define and Fit Model
values = reframed.values
n_train = int(len(values) - n_test)

train = values[:n_train, :]

test = values[n_train:, :]

file_path = 'mse_multi_var_nino34.h5'
if not os.path.exists(file_path):
    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    model.save(file_path)
else:
    model = load_model(file_path)


forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

# evaluate forecasts
actual = [row[-n_seq:] for row in test]
evaluate_forecasts(actual, forecasts, n_lag, n_seq)

# plot forecasts
# print(df['soi'].values)
# print(len(values))
# plot_forecasts(df['NINO3_4'].values, forecasts, test.shape[0] + n_seq - 1, 0, len(values), n_seq)
plot_forecasts(df['NINO3_4'][-96:].values, forecasts, n_test+11)


