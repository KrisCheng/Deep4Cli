#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the baseline model of nino index prediction.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Reference: https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
Author: Kris Peng
Date: 2018/05/16
'''

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

series = read_csv('../../data/oni/csv/nino3_4_anomaly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
# pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()