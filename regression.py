#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the prediction of NINO3.4
Author: Kris Peng
Copyright (c) 2017 - Kris Peng <kris.dacpc@gmail.com>
'''

from iostream import read as read
from preprocess.generating_targets import regression_set
from learning.regression import keras_ANN
import pandas as pd
import matplotlib.pyplot as plt

path = "data/NINO3.txt"
NINO = read.read_csv(path, sep="\t", date_key='date_time')

path = "data/ST_windburst.txt"
STwind = read.read_csv(path=path, sep='\t', date_key='date_time')

raw_data = pd.concat([NINO, STwind], axis=1).dropna(axis=0)

print(raw_data)

X, y = regression_set(raw_data, target_key='NINO3', initial_time=1969, horizon=1)

model = keras_ANN.KerasRegressionModel(arity=3, network_structure=(5, 1), batch_size=1, nb_epoch=1)
model.fit(X, y)
yhat = model.predict(X)

# plot the testdata
path = "data/nino_3_4_10d_test.txt"
NINOTest = read.read_csv(path, sep="\t", date_key='date_time')
plt.plot(range(len(NINOTest)), NINOTest, '--', color='red', label="actual")
plt.legend(loc='upper left')

plt.plot(range(len(yhat)), yhat, '-', color='blue', label="predict")
# plt.plot(range(len(y)), y, '--', color='red', label="actual")
# plt.legend(loc='upper left')

# calculate the cost(NRMSE)
cost = 0
# for n in range(len(NINOTest)):
#     cost += (NINOTest[n][0] - y[n])**2
#
# cost = (1/(max(NINOTest) - min(y))) * ((cost/len(y))**0.5)
print(cost)
plt.show()

