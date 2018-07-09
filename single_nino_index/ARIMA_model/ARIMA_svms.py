#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  apply the ARIMA model, on NINO 3.4 Index monthly data.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Reference: https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
Author: Kris Peng
Date: 2018/07/09
'''

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import boxcox
from matplotlib import pyplot
import numpy

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
# series = Series.from_csv('dataset.csv')
# groups = series['1870':'1987'].groupby(TimeGrouper('A'))
# years = DataFrame()
# for name, group in groups:
# 	years[name.year] = group.values
# years.boxplot()
# pyplot.show()
# pyplot.figure()
# pyplot.subplot(211)
# plot_acf(series, ax=pyplot.gca())
# pyplot.subplot(212)
# plot_pacf(series, ax=pyplot.gca())
# pyplot.show()

series = Series.from_csv('dataset.csv')
print(series.describe())
# prepare data
X = series.values
X = X.astype('float32')
# difference data
months_in_year = 12
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(0,0,1))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = -0.128517
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])


# # prepare data
# X = series.values
# X = X.astype('float32')
# # difference data
# months_in_year = 12
# diff = difference(X, months_in_year)
# # fit model
# model = ARIMA(diff, order=(0,0,1))
# model_fit = model.fit(trend='nc', disp=0)
# # bias constant, could be calculated from in-sample mean residual

# bias = 0
# # save model
# model_fit.save('model.pkl')
# numpy.save('model_bias.npy', [bias])