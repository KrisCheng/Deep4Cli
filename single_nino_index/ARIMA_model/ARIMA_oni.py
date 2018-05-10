#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  apply the ARIMA model, on the oni index anomaly dataset.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

raw_path = '../data/oni/csv/nino3_4_anomaly.csv'

import warnings
from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from matplotlib import pyplot
import numpy
# load data

series = Series.from_csv('dataset.csv')
# data visulization
# series.plot()
# pyplot.show()

# pyplot.figure(1)
# pyplot.subplot(211)
# series.hist()
# pyplot.subplot(212)
# series.plot(kind='kde')
# pyplot.show()

# create a differenced series
def difference(dataset, interval=1):
diff = list()
for i in range(interval, len(dataset)):
    value = dataset[i] - dataset[i - interval]
    diff.append(value)
return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# Grid Search to get the suitable hyper parameters (Resource Insufficient)
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
# def evaluate_arima_model(X, arima_order):
#   # prepare training dataset
#   X = X.astype('float32')
#   train_size = int(len(X) * 0.50)
#   train, test = X[0:train_size], X[train_size:]
#   history = [x for x in train]
#   # make predictions
#   predictions = list()
#   for t in range(len(test)):
#     # difference data
#     months_in_year = 12
#     diff = difference(history, months_in_year)
#     model = ARIMA(diff, order=arima_order)
#     model_fit = model.fit(trend='nc', disp=0)
#     yhat = model_fit.forecast()[0]
#     yhat = inverse_difference(history, yhat, months_in_year)
#     predictions.append(yhat)
#     history.append(test[t])
#   # calculate out of sample error
#   rmse = sqrt(mean_squared_error(test, predictions))
#   return rmse
# # evaluate combinations of p, d and q values for an ARIMA model
# def evaluate_models(dataset, p_values, d_values, q_values):
#     dataset = dataset.astype('float32')
#     best_score, best_cfg = float("inf"), None
#     order = (0,0,4)
# #   for p in p_values:
# #     for d in d_values:
# #      for q in q_values:
# #        order = (p,d,q)
#     try:
#         rmse = evaluate_arima_model(dataset, order)
#         if rmse < best_score:
#             best_score, best_cfg = rmse, order
#         print( 'ARIMA%s RMSE=%.2f'  % (order,rmse))
#     except:
#         pass
#     print( 'Best ARIMA%s RMSE=%.2f'  % (best_cfg, best_score))
# # load dataset
# series = Series.from_csv('dataset.csv')
# # evaluate parameters
# p_values = range(0, 7)
# d_values = range(0, 3)
# q_values = range(0, 7)
# warnings.filterwarnings("ignore")
# evaluate_models(series.values, p_values, d_values, q_values)

# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
# load data
series = Series.from_csv('dataset.csv')
# prepare data
X = series.values
X = X.astype('float32')
# difference data
months_in_year = 12
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(1,0,3))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = 0
# save model
model_fit.save("model.pkl")
numpy.save("model_bias.npy", [bias])