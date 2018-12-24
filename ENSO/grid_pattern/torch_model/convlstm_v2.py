# TODO:
# Ref: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import numpy as np
from math import sqrt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        # print(input_tensor.shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # print(combined.shape)
        combined_conv = self.conv(combined)
        # print(combined_conv.shape)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # if not self.batch_first:
        # (t, b, c, h, w) -> (b, t, c, h, w)

        # print(input_tensor.shape)
        # input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
        # print(input_tensor.shape)

        # # Implement stateful ConvLSTM
        # if hidden_state is not None:
        #     raise NotImplementedError()
        # else:
        hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

input_size = (10, 50)
input_dim = 1
hidden_dim = 1
kernel_size = (5, 5)
num_layers = 10
lr = 0.1
num_epochs = 1000
train_length = 1800
len_seq = 1980
len_frame = 12
start_seq = 1801
end_seq = 1968

def mse_loss(input, target):
    return (torch.sum((input - target) ** 2)/(500))

if __name__ == '__main__':
    data = torch.load('sst+1.pt')
    train_data = data[0]
    test_data = data[1]
    train_data = train_data.permute(0, 1, 4, 2, 3)
    test_data = test_data.permute(0, 1, 4, 2, 3)
    train_data = Variable(train_data).cuda()
    test_data = Variable(test_data).cuda()

    print("trainset shape: ", train_data.shape)
    print("testset shape: ", test_data.shape)

    conv_lstm = ConvLSTM(input_size = input_size,
                        input_dim = input_dim,
                        hidden_dim = hidden_dim,
                        kernel_size = kernel_size,
                        num_layers = num_layers)

    conv_lstm = nn.DataParallel(conv_lstm)
    conv_lstm.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(conv_lstm.parameters(), lr=lr)

    for i in range(num_epochs):
        inputs = train_data[:train_length]
        targets = test_data[:train_length]
        outputs = conv_lstm(input_tensor=inputs)
        outputs = outputs[0][0]
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: {} , Loss: {:.4f}".format(i+1, loss.item()))
    torch.save(conv_lstm.state_dict(), 'model.ckpt')

    model_sum_rmse = 0
    base_sum_rmse = 0

    for k in range(start_seq, end_seq):
        # rolling-forecasting with -n steps
        model_sum_rmse_current = 0
        base_sum_rmse_current = 0

        pred_sequence_raw = train_data[k][::, ::, ::, ::]
        pred_sequence = train_data[k][::, ::, ::, ::]
        act_sequence = train_data[k+len_frame][::, ::, ::, ::]

        for j in range(len_frame):
            new_frame = conv_lstm(input_tensor=pred_sequence[np.newaxis, ::, ::, ::, ::])
            new_frame = new_frame[0][0]
            new = new_frame[::, -1, ::, ::, ::]
            pred_sequence = torch.cat((pred_sequence, new),0)

            baseline_frame = pred_sequence_raw[j, 0, ::, ::]
            pred_toplot = pred_sequence[-1, 0, ::, ::]
            act_toplot = act_sequence[j, 0, ::, ::]
            pred_sequence = pred_sequence[1:len_frame+1, ::, ::, ::]

            model_rmse = mse_loss(act_toplot, pred_toplot)
            baseline_rmse = mse_loss(act_toplot, baseline_frame)

            model_sum_rmse, base_sum_rmse = model_sum_rmse + model_rmse, base_sum_rmse + baseline_rmse
            model_sum_rmse_current, base_sum_rmse_current = model_sum_rmse_current + model_rmse, base_sum_rmse_current + baseline_rmse

        # print("="*10)
        # print("Round: %s" % str(k+1))
        # print("Total Model RMSE: %s" % (sqrt(model_sum_rmse_current/len_frame)))
        # print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse_current/len_frame)))

    print("="*20)
    print("Total Model RMSE: %s" % (sqrt(model_sum_rmse/(len_frame*(end_seq-start_seq)))))
    print("Total Baseline RMSE: %s" % (sqrt(base_sum_rmse/(len_frame*(end_seq-start_seq)))))
