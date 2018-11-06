'''
Desc: data analysis and preprocessing.
Author: Kris Peng
Data: https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html
Data Desc: 1850/01~2015/12; monthly SST
           1.0 degree latitude x 1.0 degree longitude global grid (360x180).
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

# data preprocessing
sst = '../../../../dataset/sst_grid_1/convert_sst.mon.mean_185001_201512.mat'
sst = 'convert_sst.mon.mean_1850_01_2015_12.mat'
sst_data = sio.loadmat(sst)
sst_data = sst_data['sst'][:,:,:]
sst_data = np.array(sst_data, dtype=float)


# (180 * 360 * 2004) --> (10 * 50 * 2004) / NINO3.4 region (5W~5N, 170W~120W)
sst_data = sst_data[85:95, 190:240, :]
print("Data Shape: %s" % str(sst_data.shape))
print("Mean: %s" % sst_data.mean())
print("Max: %s , Loc: %s" % (sst_data.max(), sst_data.argmax()))
print("Min: %s , Loc: %s" % (sst_data.min(), sst_data.argmin()))

# trend analysis
x = []
y = []
sst_avg = []
for i in range(2004):
    x.append(i)
    sst_avg.append(np.mean(sst_data[:,:,i]))
    # fig = plt.figure(figsize=(10, 4))
    # plt.imshow(sst_data[::,::,i])
    # cbar = plt.colorbar(plt.imshow(sst_data[::,::,i]), orientation='horizontal')
    # cbar.set_label('°C',fontsize=12)
    # plt.savefig('%i_sst.png' % i)

fig = plt.figure(figsize=(20, 5))
plt.title("All monthly SST(1850.01~2015.12)")
plt.xlabel("month")
plt.ylabel("SST")
plt.plot(x[:-12*7],sst_avg[:-12*7])
plt.plot(x[-12*7:],sst_avg[-12*7:])
plt.savefig('sst_plot.png')
