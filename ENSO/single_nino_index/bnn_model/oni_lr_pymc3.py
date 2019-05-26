#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the LR model of nino index prediction, implemented by PyMC3.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Author: Kris Peng
Date: 2019/05/12
'''

import os
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pymc3 as pm
import theano
from theano import shared
from scipy import stats, optimize

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
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        # value = dataset[i]
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    raw_values = series.values

	# # transform data to be stationary
	# diff_series = difference(raw_values, 1)
	# diff_values = diff_series.values
	# diff_values = diff_values.reshape(len(diff_values), 1)

	# rescale values to -1, 1
	# scaler = MinMaxScaler(feature_range=(-1, 1))
	# scaled_values = scaler.fit_transform(diff_values)
	# scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	# supervised_values = supervised.values
	# train, test = supervised_values[0:-n_test], supervised_values[-n_test:]

    train = []
    test = []
    for i in range(len(series) - n_test):
        train_list  = []
        for j in range(0, n_lag + n_seq):
            train_list.append(raw_values[i + j])
        train.append(train_list)
    for i in range(len(series) - n_test, len(series) - n_lag):
        test_list  = []
        for j in range(0, n_lag + n_seq):
            test_list.append(raw_values[i + j])
        test.append(test_list)
    return np.array(train), np.array(test)

# configure
n_lag = 12
n_seq = 1
n_test = 108

series = read_csv('../../data/oni/csv/nino3_4_anomaly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# print(series.values)
train, test = prepare_data(series, n_test, n_lag, n_seq)
train_X, train_Y = train[::,0:12], train[::,12]
test_X, test_Y = test[::,0:12], test[::,12]

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

#Preprocess data for Modeling
shA_X = shared(train_X)
#Generate Model
linear_model = pm.Model()

with linear_model:
    alpha = pm.Normal("alpha", mu=0, sd=1)
    betas = pm.Normal("betas", mu=0, sd=10, shape=12)
    sigma = pm.HalfNormal("sigma", sd=1)

    # Expected value of outcome
    mu = alpha + np.array([betas[j] * shA_X[:,j] for j in range(12)]).sum()

    # Likelihood (sampling distribution of observations)
    likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=train_Y)

    # Obtain starting values via Maximum A Posteriori Estimate
    map_estimate = pm.find_MAP(model=linear_model, fmin=optimize.fmin_powell)

    # Instantiate Sampler
    step = pm.NUTS(scaling=map_estimate)

    # MCMC
    trace = pm.sample(1000, step, start=map_estimate, progressbar=True, njobs=1)

#Traceplot
pm.traceplot(trace)

# Prediction
shA_X.set_value(train_X)
ppc = pm.sample_ppc(trace, model=linear_model, samples=1000)

# What's the shape of this? 
print(list(ppc.items())[0][1].shape) #(1000, 111) it looks like 1000 posterior samples for the 111 test samples (X_te) I gave it

predict = []
actual = []
for idx in range(96):
    predicted_yi = list(ppc.items())[0][1].T[idx].mean()
    actual_yi = test_Y[idx]
    predict.append(predicted_yi)
    actual.append(actual_yi)
    # print(predicted_yi, actual_yi)
fig = plt.figure()
x = [i for i in range(96)]
plt.plot(x, predict, '-g', label="prediction")
plt.plot(x, actual, '-.k', label="actual")
sum = 0
for i in range(96):
    sum = sum + (predict[i] - actual[i]) ** 2
plt.title('Bayesian Regression Result')
plt.ylabel("NINO 3.4 Index")
plt.show()
print("RMSE: " + str((sum/96) ** 0.5))