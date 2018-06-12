#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the baseline model of nino index prediction.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Reference: https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
Author: Kris Peng
Date: 2018/05/16
'''

from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

# convert time series into supervised learning problem
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
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test

# make a persistence forecast
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]

# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    sum_rmse = []
    for i in range(n_seq):
        actual = test[:,(n_lag+i)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        sum_rmse.append(rmse)
        # print('t+%d RMSE: %f' % ((i+1), rmse))
        print("%.3f" % rmse)
    # print(len(sum_rmse))
    print("%.3f" % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5])/6))
    print("%.3f" % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5]+sum_rmse[6]+sum_rmse[7]+sum_rmse[8])/9))
    print("%.3f" % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5]+sum_rmse[6]+sum_rmse[7]+sum_rmse[8]+sum_rmse[9]+sum_rmse[10]+sum_rmse[11])/12))
    # print('6 month RMSE Avg: %f' % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5])/6))
    # print('9 month RMSE Avg: %f' % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5]+sum_rmse[6]+sum_rmse[7]+sum_rmse[8])/9))
    # print('12 month RMSE Avg: %f' % ((sum_rmse[0]+sum_rmse[1]+sum_rmse[2]+sum_rmse[3]+sum_rmse[4]+sum_rmse[5]+sum_rmse[6]+sum_rmse[7]+sum_rmse[8]+sum_rmse[9]+sum_rmse[10]+sum_rmse[11])/12))

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values, label='observed')
	pyplot.legend(loc='upper right')
	pyplot.title("oni_baseline_timeseries")
	# plot the forecasts in red
	for i in range(len(forecasts)):
		if i%n_seq == 0 and i != 0:
		    off_s = len(series) - n_test + i - 1
		    off_e = off_s + len(forecasts[i]) + 1
		    xaxis = [x for x in range(off_s, off_e)]
		    yaxis = [series.values[off_s]] + forecasts[i]
		    pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

series = read_csv('../../data/oni/csv/nino3_4_anomaly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# configure
n_lag = 1
n_seq = 12
n_test = 360
# prepare data
train, test = prepare_data(series, n_test, n_lag, n_seq)

# make forecasts
forecasts = make_forecasts(train, test, n_lag, n_seq)
# evaluate forecasts

evaluate_forecasts(test, forecasts, n_lag, n_seq)
# plot forecasts
# print(forecasts)
plot_forecasts(series[-n_test:], forecasts, n_test+11)