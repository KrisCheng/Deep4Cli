#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  apply the LSTM model for one-step univariate time series forecasting, on the oni index anomaly dataset, based on keras (from 1870~2017, monthly data).
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Reference: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
Author: Kris Peng
Copyright (c) 2017 - Kris Peng <kris.dacpc@gmail.com>
'''

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
import scipy.io as sio   
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from pandas import Series
from math import sqrt
import numpy,pandas

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag = 1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag + 1)]
	columns.append(df)
	df = concat(columns, axis = 1)
	df.fillna(0, inplace = True)
	return df
 
# create a differenced series
def difference(dataset, interval = 1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval = 1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):	
	# fit scaler
	scaler = MinMaxScaler(feature_range = (-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape = (batch_size, X.shape[1], X.shape[2]), stateful = True))
	model.add(Dense(1))
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	print(model.summary())
	# TODO: why?? (同一个模型多次训练？？而且每次仅仅一轮)
	# for i in range(nb_epoch): 
		# model.fit(X, y, epochs = 1, batch_size = batch_size, verbose = 1, shuffle = False)
		# model.reset_states()
	history = model.fit(X, y, epochs = nb_epoch, batch_size = batch_size, verbose = 1, shuffle = False)
	pyplot.plot(history.history['loss'])
	pyplot.title('model loss')
	pyplot.ylabel('loss')
	pyplot.xlabel('epoch')
	pyplot.legend(['train', 'test'], loc='upper left')
	pyplot.show()

	# save the trained model
	# model.save('100e_mse_nino34_model.h5')

	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size = batch_size)
	return yhat[0,0]

# load data
# from txt
# raw = '../data/oni/txt/nino4_anomaly.txt'
# raw_data = numpy.loadtxt(raw)
# raw_data = numpy.delete(raw_data, 0, 1)
# raw_values = raw_data.reshape(1, 12 * 148)
# raw_values = raw_values[0]

# from csv
raw = '../../data/oni/csv/nino4_anomaly.csv'
series = pandas.read_csv(raw, header=0, parse_dates=[0], index_col=0, squeeze=True)
# transform to supervised learning
raw_values = series.values

# transform data to be stationary
raw_values = raw_values
# print(raw_values)

diff_values = difference(raw_values, 1)
# print(diff_values)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
# print(supervised_values)

# split data into train and test-sets
train, test = supervised_values[0:-228], supervised_values[-228:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
 
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 10, 40)

# load the exist model
# lstm_model = load_model('100e_mse_nino34_model.h5')

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size = 1)
 
time = []
currentYear = 1999
currentMonth = 1

# walk-forward validation on the test data
predictions = list()

for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
	# invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]

	# time calculates
    if(currentMonth is 13):
        currentYear = currentYear + 1
        currentMonth = 1
    temp = str(str(currentYear) + '/' + str(currentMonth))
    time.append(temp)
    currentMonth = currentMonth + 1
    print('Month=%s, Predicted=%f, Expected=%f' % (temp, yhat, expected))

# report performance
mse = mean_squared_error(raw_values[-228:], predictions)
print('Test MSE: %.3f' % mse)
# line plot of observed vs predicted
xs = [datetime.strptime(t, '%Y/%m').date() for t in time]
pyplot.plot(xs, raw_values[-228:], color = "blue", label = "actual")
pyplot.plot(xs, predictions, color = "red", linestyle = '--', label = "predict")
pyplot.legend(loc = 'upper left')
pyplot.xlabel('time(years)')
pyplot.ylabel('NINO3.4/°C')
pyplot.show()