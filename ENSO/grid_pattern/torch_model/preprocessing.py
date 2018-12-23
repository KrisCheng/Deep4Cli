import numpy as np
import torch

np.random.seed(2)
DATA_PATH = 'monthly_sst+1.npy'
train_X_raw, train_Y_raw, sst_grid_raw = np.load(DATA_PATH) # from .npy file
data = [train_X_raw, train_Y_raw]
data = torch.tensor(data, dtype=torch.float)
torch.save(data, open('sst+1.pt', 'wb'))
