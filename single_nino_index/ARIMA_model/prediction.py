#!/usr/bin/python
# -*- coding: utf-8 -*-

# load and evaluate the finalized model on the validation dataset
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from pandas import datetime
from math import sqrt
import numpy
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
# load and prepare datasets
dataset = Series.from_csv('dataset.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
validation = Series.from_csv('validation.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')

# make first prediction
predictions = list()
# yhat = float(model_fit.forecast()[0])
# yhat = bias + inverse_difference(history, yhat, months_in_year)
# predictions.append(yhat)
# history.append(y[0])
# print('>Predicted=%.2f, Expected=%.2f' % (yhat, y[0]))
# rolling forecasts
time = []
currentYear = 1989
currentMonth = 1

for i in range(0, len(y)):
    # difference data
    months_in_year = 12
    diff = difference(history, months_in_year)
    # predict
    model = ARIMA(diff, order=(0,0,1))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    yhat = bias + inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
    if(currentMonth is 13):
        currentYear = currentYear + 1
        currentMonth = 1
    temp = str(str(currentYear) + '/' + str(currentMonth))
    time.append(temp)
    currentMonth = currentMonth + 1

#   print('>Predicted=%.2f, Expected=%.2f'  % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
# pyplot.plot(y)
# pyplot.plot(predictions, color='red')
xs = [datetime.strptime(t, '%Y/%m').date() for t in time]
pyplot.plot(xs, y, color = "blue", label = "actual")
pyplot.plot(xs, predictions, color = "red", linestyle = '--', label = "predict")
pyplot.legend(loc = 'upper left')
pyplot.xlabel('time(years)')
pyplot.ylabel('NINO3.4/°C')
pyplot.show()