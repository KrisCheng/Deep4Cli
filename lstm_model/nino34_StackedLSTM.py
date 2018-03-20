'''
Desc: the Stacked LSTM model for NINO3.4 index, based on keras.(Unfinished)
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/nino34.long.anom.data
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

raw = '../data/nino34/nino3_4_anomaly.txt'  
raw_data = numpy.loadtxt(raw)
raw_data = numpy.delete(raw_data, 0, 1)
raw_values = raw_data.reshape(1, 12 * 147)
raw_values = raw_values[0]

# generate damped sine wave [0, 1]
def generate_sequence(length, period, decay):
    return [0.5 + 0.5 * sin(2 * pi * i / period) * exp(-decay * i) for i in range(length)]

# generate input-output pairs
def generate_examples(length, n_patterns, output):
    X, y = list(), list()
    for _ in range(n_patterns):
        p = randint(10, 20)
        d = uniform(0.01, 0.1)
        
        sequence = generate_sequence(length + output, p, d)
        sequence = raw_values
        print(len(sequence))
        X.append(sequence[:-output])
        y.append(sequence[-output:])
    # X = array(X).reshape(len(X), 1, 1)
    # y = array(y).reshape(len(y), output)
    return X, y

# configure number
length = 1536
output = 228

# define model
model = Sequential()
model.add(LSTM(20, return_sequences = True, input_shape = (length, 1)))
model.add(LSTM(20))
model.add(Dense(output))
model.compile(loss = 'mae', optimizer = 'adam')
print(model.summary())

# data prepare 
# train: 1870.01~1997.12 (1536)
# test: 1998.01~2016.12 (228)

# print(raw_values)
X, y = list(), list()
X = raw_values[:-228]
y = raw_values[-228:]
# X, y = generate_examples(length, 1, output)
X = array(X).reshape(1, 1536, 1)
y = array(y).reshape(1, 228)
# print(X)

# fit model
history = model.fit(X, y, batch_size = 20, epochs = 100)

# evaluate model
# X, y = generate_examples(length, 100, output)
loss = model.evaluate(X, y, verbose = 0)
print("MAE: %f" % loss)

# predict new data
# X, y = generate_examples(length, 1, output)
yhat = model.predict(X, verbose = 0)

time = []
currentYear = 1998
currentMonth = 1

for i in range(228):
    if(currentMonth is 13):
        currentYear = currentYear + 1
        currentMonth = 1
    temp = str(str(currentYear) + '/' + str(currentMonth))
    time.append(temp)
    currentMonth = currentMonth + 1

xs = [datetime.strptime(t, '%Y/%m').date() for t in time]
pyplot.plot(xs, y[0], color = "blue", label = "actual")
pyplot.plot(xs, yhat[0], color = "red", linestyle = '--', label = "predict")
pyplot.legend(loc = 'upper left')

pyplot.legend()
pyplot.show()