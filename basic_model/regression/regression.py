#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the prediction of NINO3.4, a regression problem.
Author: Kris Peng
Copyright (c) 2017 - Kris Peng <kris.dacpc@gmail.com>
'''

from iostream import read as read
from preprocess.generating_targets import regression_set
import keras_ANN
import pandas as pd
import matplotlib.pyplot as plt

path = "../../data/NINO34.txt"
NINO = read.read_csv(path, sep="\t", date_key='date_time')

path = "../../data/Sd_PC2.txt"
STwind = read.read_csv(path=path, sep='\t', date_key='date_time')

raw_data = pd.concat([NINO, STwind], axis=1).dropna(axis=0)

X, y = regression_set(raw_data, target_key='NINO3', initial_time=2003.5, horizon=1)
model = keras_ANN.KerasRegressionModel(arity=3, network_structure=(5, 1), batch_size=1, nb_epoch=200)
model.fit(X, y)
yhat = model.predict(X)

# calculate the cost(NRMSE)
cost = 0
for n in range(len(y)):
    cost += (y[n] - yhat[n])**2
cost = (1/(max(y) - min(yhat)) * ((cost/len(yhat))**0.5))
print("NRMSE: %f" %cost)

# plot
plt.plot(range(len(y)), y, '--', color='red', label="actual")
plt.plot(range(len(yhat)), yhat, '-', color='blue', label="predicted")
plt.legend(loc='upper right')
plt.show()

