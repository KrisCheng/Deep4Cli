'''
Desc: the construction of dataset from .nc file.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''

import pylab as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from netCDF4 import Dataset

# # === daily ssta
# 1.读取数据并截取nino34区域，存储为.npy
# for year in range(1982, 2018):
#     file_path = "ssta_0.25_0.25/sst.day.anom."+str(year)+".nc"
#     nc = Dataset(file_path)
#     sst_data = nc["anom"][::,::,::]
#     sst_data = np.array(sst_data, dtype=float)
#     # print(sst_data.shape)
#     converted_data = []
#     # 转化数据
#     for day_ssta in sst_data:
#         con_data = []
#         for line in day_ssta:
#             con_data.insert(0, line)
#         converted_data.append(con_data)
#     converted_data = np.array(converted_data, dtype=float)    
#     plt.imshow(converted_data[0,::,::])
#     # cbar = plt.colorbar(plt.imshow(converted_data[0,::,::]), orientation='horizontal')
#     plt.savefig('data/%i.png' % year)
#     converted_data = converted_data[::,85*4:95*4,190*4:240*4]
#     print("year: %i" % year)
#     print(converted_data.shape)
#     print("*"*10)
#     np.save("data/datassta_"+str(year)+".npy", converted_data)


# # 2. 处理闰年 31+28 = 59 --> [59]
# for year in range(1984,2017,4):
#     r = np.load("data/ssta_"+str(year)+".npy")
#     r = np.delete(r,59,axis=0)
#     print(r.shape)
#     np.save("data/ssta_"+str(year)+".npy", r)

# # 3.合并成一个文件
# all_daily_ssta = []
# for year in range(1982,2018):
#     yearly_ssta = np.load("data/ssta_"+str(year)+".npy")
#     yearly_ssta = yearly_ssta.tolist()
#     all_daily_ssta.append(yearly_ssta)
# all_daily_ssta = np.array(all_daily_ssta, dtype=float)
# np.save("data/ssta_daily_1982_2017_nino34.npy", all_daily_ssta)
# print(all_daily_ssta.shape)


# monthly sst
DATA_PATH = "convert_sst.mon.mean_1850_01_2015_12.mat"

# load data
sst_data = sio.loadmat(DATA_PATH)
sst_data = sst_data['sst'][::,::,::]
sst_data = sst_data[85:95,190:240,::]
sst_data = np.array(sst_data, dtype=float)
all_sst = np.zeros((24,12,10,50,1), dtype=np.float)

# [0,12]->[1,13]
for i in range(24):
    for k in range(12):
        all_sst[i,k,::,::,0] = sst_data[::,::,i+k+1969]
print(all_sst.shape)
np.save("enso_monthly_sst_last_24.npy", [all_sst])