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
from pandas import datetime
from pandas import read_csv
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot

def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y-%m')
    else:
       return datetime.strptime(x, '%Y-0%m') 

# load dataset
# df = read_csv('../data/oni/csv/all_nino_anomaly.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)

series = Series.from_csv('../data/oni/csv/nino3_4_anomaly.csv', header=0)
# # 1.Bacic information
print(series.describe())
lag_plot(series)
pyplot.show()

# # print(dataset.head())
# values = df.values
# # specify columns to plot
# groups = [0, 1, 2, 3]
# i = 1
# # plot each column
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups), 1, i)
# 	pyplot.hist(values[:, group])
# 	pyplot.title(df.columns[group], y=0.6, loc='right')
# 	if i != 4:
# 		pyplot.tick_params(
# 			axis='x',          # changes apply to the x-axis
# 			which='both',      # both major and minor ticks are affected
# 			bottom='off',      # ticks along the bottom edge are off
# 			top='off',         # ticks along the top edge are off
# 			labelbottom='off') # labels along the bottom edge are off
# 		# pyplot.xticks([], [])
# 		pyplot.xlabel('')
# 	i += 1
# pyplot.show()

# # 1. line plot
# i = 1
# fig = pyplot.figure()
# for col in df.columns.tolist():
#     fig.add_subplot(len(df.columns.tolist()), 1, i)
#     df[col].plot()
#     # pyplot.plot(values[:, i-1])
#     pyplot.title(col, y=0.7, loc='right')
#     if i != len(df.columns.tolist()):
#     # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
#         pyplot.tick_params(
#             axis='x',          # changes apply to the x-axis
#             which='both',      # both major and minor ticks are affected
#             bottom='off',      # ticks along the bottom edge are off
#             top='off',         # ticks along the top edge are off
#             labelbottom='off') # labels along the bottom edge are off
#         # pyplot.xticks([], [])
#         pyplot.xlabel('')
#     i += 1
# pyplot.show()



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

# # 4.Box and Whisker
# groups = series['2010':'2017'].groupby(TimeGrouper('A'))
# years = DataFrame()
# for name, group in groups:
# 	years[name.year] = group.values
# years.boxplot()
# pyplot.show()
