#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the LR model of nino index prediction, implemented by Pyro.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Author: Kris Peng
Date: 2019/05/12
'''

from functools import partial
import seaborn as sns
import torch
import torch.nn as nn
from pandas import Series

import os
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import os.path

class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x).unsqueeze(1)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	# scaled_values = diff_values
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

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

p = 2  # number of features
regression_model = RegressionModel(p)

loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)
num_iterations = 5000

DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/rugged_data.csv"
# configure
n_lag = 0
n_seq = 2
n_test = 96

# data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
series = pd.read_csv('../../data/oni/csv/nino3_4_anomaly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
train_1 = series.data[:-2]
train_2 = series.data[1:-1]
test_1 = series.data[2:]
train = []
test = []
for i in range(1774):
    list1 = [train_1[i], train_2[i]]
    list2 = [test_1[i]]
    train.append(list1)
    test.append(list2)
test = np.asarray(train)
train = torch.tensor(train, dtype=torch.float) 
test = torch.tensor(test, dtype=torch.float)

data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
data = torch.tensor(df.values, dtype=torch.float)

def main():
    # x_data = data[:, :-1]
    # y_data = data[:, -1]
    x_data = train
    y_data = test

    # print(train.shape)
    # print(test.shape)
    # print(data.shape)
    print(x_data.shape)
    print(y_data.shape)
    for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print(name, param.data.numpy())
main()
