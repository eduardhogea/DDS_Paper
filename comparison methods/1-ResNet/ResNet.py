import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish_act(nn.Module):
    def __init__(self):
        super(Swish_act, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            Swish_act(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.Lrelu = Swish_act()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.Lrelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel=33, num_class=10):    # batch_size, 33, 1000
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channel, 64, kernel_size=7, stride=1, padding=3),  # 1, 64, 1000
            nn.BatchNorm1d(64),
            Swish_act(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # batch, 64, 500
        )

        self.layer1 = ResBlock(64, 96, stride=1)  # batch, 96, 500
        self.layer2 = ResBlock(96, 128, stride=2)  # batch, 128, 250
        self.layer3 = ResBlock(128, 192, stride=2)  # batch, 192, 125
        self.layer4 = ResBlock(192, 256, stride=2)  # batch, 256, 63
        self.layer5 = ResBlock(256, 384, stride=2)  # batch, 384, 32

        self.linear = nn.Sequential(
            nn.Linear(384 * 32, 100),
            nn.Dropout(),
            nn.Linear(100, num_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        pred = self.linear(out.view(out.size(0), -1))
        return pred


if __name__ == '__main__':
    model = ResNet(num_class=20)
    x = torch.randn(64, 33, 1000)
    y = model(x)
    print(y.shape)
