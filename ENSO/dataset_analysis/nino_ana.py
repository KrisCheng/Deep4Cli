from pandas import Series
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from random import randrange
from pandas import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import numpy

series = read_csv('../data/oni/csv/nino3_4_anomaly.csv', header=0, parse_dates=[0], index_col=0,
    squeeze=True)

X = series.values
result = adfuller(X)
print('ADF Statistic: %f'  % result[0])
print('p-value: %f'  % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print( '\t%s: %.3f'  % (key, value))

# fit linear model
# X = [i for i in range(0, len(series))]
# X = numpy.reshape(X, (len(X), 1))
# y = series.values
# model = LinearRegression()
# model.fit(X, y)
# # calculate trend
# trend = model.predict(X)
# # plot trend
# pyplot.plot(y)
# pyplot.plot(trend)
# pyplot.show()
# # detrend
# for i in trend:
#     print(i)
# detrended = [y[i]-trend[i] for i in range(0, len(series))]
# # plot detrended
# pyplot.plot(detrended)
# pyplot.show()
# # # detrend
# # detrended = [y[i]-trend[i] for i in range(0, len(series))]
# # # plot detrended
# # pyplot.plot(detrended)
# # pyplot.show()