'''
Desc: the oni index analysis.
DataSource: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import numpy
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot

series = Series.from_csv('../data/oni/csv/nino3_4_anomaly.csv')

# # 1.Bacic information
# print(series.describe())
# series.plot()
# pyplot.show()
# print(series.describe())


# # 2.Seasonal Line Plots
# groups = series['1870':'1877'].groupby(TimeGrouper('A'))
# years = DataFrame()
# pyplot.figure()
# i = 1
# n_groups = len(groups)
# for name, group in groups:
# 	pyplot.subplot((n_groups*100) + 10 + i)
# 	i += 1
# 	pyplot.plot(group)
# pyplot.show()

# 3.Density plot
# pyplot.figure(1)
# pyplot.subplot(211)
# series.hist()
# pyplot.subplot(212)
# series.plot(kind='kde')
# pyplot.show()

# 4.Box and Whisker
groups = series['2010':'2017'].groupby(TimeGrouper('A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
pyplot.show()
