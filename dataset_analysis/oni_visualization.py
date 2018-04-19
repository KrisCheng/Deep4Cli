#!/usr/bin/python
# -*- coding: utf-8 -*-


# import numpy
# from pandas import Series
# from pandas import DataFrame
# from pandas import TimeGrouper
# from matplotlib import pyplot
# from pandas.tools.plotting import lag_plot

# series = Series.from_csv('../data/oni/csv/nino3_4.csv', header=0)

# rolling = series.rolling(window=3)
# rolling_mean = rolling.mean()
# print(rolling_mean.head(10))
# # plot original and transformed dataset
# series.plot()
# rolling_mean.plot(color='red')
# pyplot.show()
# # zoomed plot original and transformed dataset
# series[:100].plot()
# rolling_mean[:100].plot(color='red')
# pyplot.show()

from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = Series.from_csv('../data/oni/csv/nino3_4.csv', header=0)
result = seasonal_decompose(series, model="multiplicative")
result.plot()
pyplot.show()