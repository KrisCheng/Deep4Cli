'''
Desc: data preprocessing handler.
Author: Kris Peng
Copyright (c) 2018 - Kris Peng <kris.dacpc@gmail.com>
'''
import scipy.io as sio
import numpy as np

# monthly sst
DATA_PATH = '../../../../dataset/sst_grid_1/convert_sst.mon.mean_185001_201512.mat'

# daily ssta
# DATA_PATH = 'ssta_daily_1982_2017_nino34.npy'

# monthly sst setting
len_year = 167
len_seq = 12
map_height, map_width = 10, 50
MAX = 31.18499947
MIN = 20.33499908
MEAN = 26.80007865

# 0~1 Normalization
def normalization(data):
    normalized_data = np.zeros((map_height, map_width), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            normalized_data[i][j] = (data[i][j]- MIN)/(MAX - MIN)
    return normalized_data

def inverse_normalization(data):
    inverse_data = np.zeros((map_height, map_width), dtype=np.float)
    for i in range(len(data)):
        for j in range(len(data[0])):
            inverse_data[i][j] = data[i][j]*(MAX - MIN) + MIN
    return inverse_data

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def load_data_convlstm_monthly(train_length):
    # load data
    sst_data = sio.loadmat(DATA_PATH)
    sst_data = sst_data['sst'][::,::,::]
    sst_data = np.array(sst_data, dtype=float)
    print("Shape of origin Dataset: ", sst_data.shape)

    # (180 * 360 * 2004) --> (10 * 50 * 2004)
    # the NINO3.4 region (5W~5N, 170W~120W)
    sst_data = sst_data[85:95,190:240,::]

    min = sst_data.min()
    max = sst_data.max()
    print('=' * 10)
    print("min:", min, "max:", max)
    # sst min:20.33 / max:31.18

    normalized_sst = np.zeros((len_year,len_seq,map_height,map_width,1), dtype = np.float)
    for i in range(len_year):
        for k in range(len_seq):
            # Year * 12 + currentMonth
            normalized_sst[i,k,::,::,0] = normalization(sst_data[::,::,i*len_seq+k])
    train_X = normalized_sst[:train_length]
    train_Y = np.zeros((train_length,len_seq,map_height,map_width,1), dtype = np.float)

    for i in range(train_length):
        for k in range(len_seq):
            if(k != len_seq-1):
                train_Y[i,k,::,::,0] = train_X[i,k+1,::,::,0]
            # 每一年的12月以下一年的1月为输出
            else:
                if(i != train_length-1):
                    train_Y[i,k,::,::,0] = train_X[i+1,0,::,::,0]
                else:
                # 最后一年的最后一月以当前帧为输出
                    train_Y[i,k,::,::,0] = train_X[i,k,::,::,0]

    print("Whole Shape: ", normalized_sst.shape)
    print("Train_X Shape: ", train_X.shape)
    print("Train_Y Shape: ", train_Y.shape)
    return normalized_sst, train_X, train_Y

def load_data_convlstm_daily(train_length):
    # load data
    ssta_data = np.load(DATA_PATH)
    min = ssta_data.min()
    max = ssta_data.max()
    print('=' * 10)
    print("min:", min, "max:", max)
    # min: -6.319999694824219 / max: 7.730000019073486

    normalized_ssta = np.zeros((len_year,len_seq,map_height,map_width,1), dtype = np.float)
    for i in range(len_year):
        for k in range(len_seq):
            # Year * 12 + currentMonth
            normalized_ssta[i,k,::,::,0] = normalization(ssta_data[i,k,::])

    train_X = normalized_ssta[:train_length]
    train_Y = np.zeros((train_length,len_seq,map_height,map_width,1), dtype = np.float)

    for i in range(train_length):
        for k in range(len_seq):
            if(k != len_seq-1):
                train_Y[i,k,::,::,0] = train_X[i,k+1,::,::,0]
            # 每一年的12月31日以下一年的1月1日为输出
            else:
                if(i != train_length-1):
                    train_Y[i,k,::,::,0] = train_X[i+1,0,::,::,0]
                else:
                # 最后一年的最后一天以当前帧为输出
                    train_Y[i,k,::,::,0] = train_X[i,k,::,::,0]
    print("Whole Shape: ", normalized_ssta.shape)
    print("Train_X Shape: ", train_X.shape)
    print("Train_Y Shape: ", train_Y.shape)
    return normalized_ssta, train_X, train_Y

# TODO
def load_data_resnet():
    # load data
    sst_data = sio.loadmat(DATA_PATH)
    sst_data = sst_data['sst'][:,:,:]
    sst_data = np.array(sst_data, dtype=float)

    # (180 * 360 * 2004) --> (10 * 50 * 2004) NINO3.4 region (5W~5N, 170W~120W)
    sst_data = sst_data[85:95,190:240,:]

    min = sst_data.min()
    max = sst_data.max()
    print('=' * 10)
    print("min:", min, "max:", max)

    # sst min:20.33 / max:31.18
    normalized_sst = np.zeros((len_year,12,map_height,map_width), dtype = np.float)
    for i in range(len_year):
        for k in range(12):
            # Year * 12 + currentMonth
            normalized_sst[i,k,::,::] = normalization(sst_data[::,::,i*12+k])
    return normalized_sst

# the evaluate function, based on RMSE, TODO
# def evaluate():
