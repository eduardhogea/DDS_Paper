import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from conv1d_rnncell import Conv1dRNNCell


class FlattenLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class Swish_act(nn.Module):
    def __init__(self):
        super(Swish_act, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class Conv1dRNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, bias, output_size, activation='tanh', num_class=10):
        super(Conv1dRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if activation == 'tanh':
            self.rnn_cell_list.append(Conv1dRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv1dRNNCell(self.hidden_size,
                                                       self.hidden_size,
                                                       self.kernel_size,
                                                       self.bias,
                                                       "tanh"))

        elif activation == 'relu':
            self.rnn_cell_list.append(Conv1dRNNCell(self.input_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(Conv1dRNNCell(self.hidden_size,
                                                   self.hidden_size,
                                                   self.kernel_size,
                                                   self.bias,
                                                   "relu"))
        else:
            raise ValueError("Invalid activation.")

        self.conv = nn.Conv1d(in_channels=self.hidden_size,
                             out_channels=self.output_size,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=self.bias)

        self.conv_classifier = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size,
                      out_channels=self.output_size,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            # nn.BatchNorm1d(self.output_size),
            Swish_act(),
            FlattenLayer(self.output_size),
            nn.Linear(self.output_size, num_class)
        )

    def forward(self, input, hx=None):
        # Shape of input（batch size, channel number = feature len, sequence len）
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, 1).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size, 1))

        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer])

        for t in range(input.size(2)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, :, t].unsqueeze(2), hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer])
                hidden[layer] = hidden_l

                # hidden[layer] = hidden_l

            outs.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = outs[-1]
        # print(out.shape)

        out = self.conv_classifier(out)

        return out


if __name__ == '__main__':
    # input size is the feature size, for example, we have 33 sensors.
    model = Conv1dRNN(input_size=33, hidden_size=50, kernel_size=1, num_layers=1, bias=True, output_size=100, num_class=20).cuda()
    x = torch.randn(64, 33, 1000).cuda()
    y = model(x)
    print(y.shape)