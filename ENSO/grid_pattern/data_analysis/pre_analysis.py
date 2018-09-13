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
import matplotlib as mpl
import scipy.io as sio
from matplotlib import pyplot

# data preprocessing
sst = '../../../../dataset/sst_grid_1/convert_sst.mon.mean_185001_201512.mat'
sst_data = sio.loadmat(sst)
sst_data = sst_data['sst'][:,:,:]
sst_data = np.array(sst_data, dtype=float)

# (180 * 360 * 2004) --> (10 * 50 * 2004) / NINO3.4 region (5W~5N, 170W~120W)
sst_data = sst_data[85:95, 190:240, :]
print("Data Shape: %s" % str(sst_data.shape))
print("Mean: %s" % sst_data.mean())
print("Max: %s , Loc: %s" % (sst_data.max(), sst_data.argmax()))
print("Min: %s , Loc: %s" % (sst_data.min(), sst_data.argmin()))

# visualization
vis_sst = sst_data[:,:,1]
print(vis_sst)
plt.imshow(vis_sst)
cbar = plt.colorbar(plt.imshow(vis_sst), orientation='horizontal')
cbar.set_ticklabels(('20','21','22','23','24','25','26','27','28','29','30','31','32'))
plt.show()


# # 1850.01~2015.01 (train)
# train_data = sst_data[::,::,0:-12]
# # 2015.01~2015.12 (test)
# test_data = sst_data[::,::,-12:]
# print(train_data.shape)
# print(test_data.shape)
