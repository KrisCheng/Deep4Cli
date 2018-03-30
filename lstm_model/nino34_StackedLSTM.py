'''
Desc: the Stacked LSTM model for NINO index, based on keras.(Unfinished)
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

from keras.models import Sequential
from keras.layers import LSTM, Dense
from numpy import array
from math import sin
from math import pi
from math import exp
from random import random
from random import randint
from random import uniform
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from matplotlib import pyplot
import numpy

raw = '../data/oni/nino3_4_anomaly.txt'  
raw_data = numpy.loadtxt(raw)
raw_data = numpy.delete(raw_data, 0, 1)
raw_values = raw_data.reshape(1, 12 * 148)
raw_values = raw_values[0]

# configure number
length = 792
output = 96

# define model (Stacked LSTM model)
model = Sequential()
model.add(LSTM(20, return_sequences = True, input_shape = (length, 1)))
model.add(LSTM(20))
model.add(Dense(output))
model.compile(loss = 'mse', optimizer = 'adam')
print(model.summary())

# define model (one LSTM layer model)
# model = Sequential()
# model.add(LSTM(10, input_shape = (length, 1)))
# model.add(Dense(output))
# model.compile(loss = 'mse', optimizer = 'adam')
# print(model.summary())

# data prepare 
# train: 1870.01~1997.12 (1536)
# test: 1998.01~2017.12 (240)

X, y = list(), list()
X = raw_values[:-888]
y = raw_values[-888:]

Xtrain = X[:-96]
Xtest = X[-96:]
ytrain = y[:-96]
ytest = y[-96:]

Xtrain = array(Xtrain).reshape(1, 792, 1)
Xtest = array(Xtest).reshape(1, 96)
ytrain = array(ytrain).reshape(1, 792, 1)
ytest = array(ytest).reshape(1, 96)


# X = array(X).reshape(1, 1536, 1)
# y = array(y).reshape(1, 240)

# fit model
history = model.fit(Xtrain, Xtest, epochs = 1000)

# evaluate model
loss = model.evaluate(ytrain, ytest, verbose = 0)
print("MSE: %f" % loss)

# predict new data
yhat = model.predict(ytrain, verbose = 0)

# plot part
time = []
currentYear = 1998+12
currentMonth = 1

for i in range(96):
    if(currentMonth is 13):
        currentYear = currentYear + 1
        currentMonth = 1
    temp = str(str(currentYear) + '/' + str(currentMonth))
    time.append(temp)
    currentMonth = currentMonth + 1

xs = [datetime.strptime(t, '%Y/%m').date() for t in time]
pyplot.plot(xs, ytest[0], color = "red", linestyle = '--', label = "actual")
pyplot.plot(xs, yhat[0], color = "blue", label = "predicted")
pyplot.legend(loc = 'upper left')
pyplot.xlabel('time(years)')
pyplot.ylabel('NINO3.4/°C')
pyplot.legend()
pyplot.show()