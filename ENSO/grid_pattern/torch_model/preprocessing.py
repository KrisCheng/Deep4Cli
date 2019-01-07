import numpy as np
import torch
from torch.autograd import Variable

MAX = 31.18499947
MIN = 20.33499908
MEAN = 26.80007865
map_height, map_width = 10, 50
len_year = 167
len_frame = 12
len_seq = 1980

# 0~1 Normalization
def normalization(data):
    normalized_data = torch.zeros((map_height, map_width), dtype=torch.float32)
    for i in range(len(data)):
        for j in range(len(data[0])):
            normalized_data[i][j] = (data[i][j]- MIN)/(MAX - MIN)
    return normalized_data

def inverse_normalization(data):
    inverse_data = torch.zeros((map_height, map_width), dtype=torch.float32)
    for i in range(len(data)):
        for j in range(len(data[0])):
            inverse_data[i][j] = data[i][j]*(MAX - MIN) + MIN
    return inverse_data

if __name__ == '__main__':

    np.random.seed(2)
    DATA_PATH = 'monthly_sst+1.npy'
    train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file

    # normalization, data for ConvLSTM Model -n ahead -5 dimension
    train_X = torch.zeros((len_seq, len_frame, 1, 10, 50), dtype=torch.float32)
    train_Y = torch.zeros((len_seq, len_frame, 1, 10, 50), dtype=torch.float32)
    for i in range(len_seq):
        print(i)
        for k in range(len_frame):
            train_X[i,k,0,::,::] = normalization(train_X_raw[i,k,::,::,0])
            train_Y[i,k,0,::,::] = normalization(train_Y_raw[i,k,::,::,0])
    train_data = Variable(train_X).cuda()
    test_data = Variable(train_Y).cuda()
    data = [train_data, test_data]

    torch.save(data, open('sst+1_12_normalized.pt', 'wb'))
