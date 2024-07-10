import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Conv1dRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True, nonlinearity="tanh"):
        super(Conv1dRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.kernel_size = kernel_size,
        self.padding = kernel_size // 2

        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = nn.Conv1d(in_channels=input_size,
                             out_channels=hidden_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)


        self.h2h = nn.Conv1d(in_channels=hidden_size,
                             out_channels=hidden_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size, input.size(2)))
            # print(hx.shape)

        hy1 = self.x2h(input)
        # print(hy1.shape)
        hy2 = self.h2h(hx)
        # print(hy2.shape)
        # hy = (self.x2h(input) + self.h2h(hx))
        hy = hy1 + hy2

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy



if __name__ == '__main__':
    x = torch.randn(64, 33, 1)
    cell = Conv1dRNNCell(input_size=33, hidden_size=50, kernel_size=1)
    h = cell(x)
    print(h.shape)
