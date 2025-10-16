from baseModel import *
import torch.nn.functional as F


class ResBlock_Bayesian_BN(BayesianNN):
    def __init__(self, input_channels, output_channels, stride, device):
        super(ResBlock_Bayesian_BN, self).__init__()
        self.conv1 = BayesianConv1d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, device=device)
        self.bn1 = nn.BatchNorm1d(output_channels).cuda()
        self.leakyrelu1 = BayesianLeakyReLU()

        self.conv2 = BayesianConv1d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.bn2 = nn.BatchNorm1d(output_channels).cuda()

        self.skip_conv = BayesianConv1d(input_channels, output_channels, kernel_size=1, stride=stride, device=device)
        self.skip_bn = nn.BatchNorm1d(output_channels).cuda()

        self.leakyrelu2 = BayesianLeakyReLU()

    def forward(self, x, kl_0):
        out, kl_1 = self.conv1(x, kl_0)
        out = self.bn1(out)
        out, kl_1 = self.leakyrelu1(out, kl_1)
        out, kl_1 = self.conv2(out, kl_1)
        out = self.bn2(out)

        skip_out, skip_kl = self.skip_conv(x, kl_0)
        skip_out = self.skip_bn(skip_out)

        out = out + skip_out
        kl_2 = kl_1 + skip_kl
        out, kl_2 = self.leakyrelu2(out, kl_2)
        return out, kl_2


class Bayes_ResNet_BN(BayesianNN):
    def __init__(self, device, num_class=10):
        super(Bayes_ResNet_BN, self).__init__()
        self.bayes_conv0 = BayesianConv1d(33, 64, kernel_size=7, stride=2, padding=3, device=device)   # bz, 36, 1000
        self.bn0 = nn.BatchNorm1d(64).cuda()
        self.leakyrelu0 = BayesianLeakyReLU()

        self.bayes_layer1 = ResBlock_Bayesian_BN(64, 96, stride=1, device=device)    # batch, 48, 500
        self.bayes_layer2 = ResBlock_Bayesian_BN(96, 128, stride=2, device=device)    # batch, 64, 500
        self.bayes_layer3 = ResBlock_Bayesian_BN(128, 192, stride=2, device=device)    # batch, 96, 250
        self.bayes_layer4 = ResBlock_Bayesian_BN(192, 256, stride=2, device=device)    # batch, 128, 125
        self.bayes_layer5 = ResBlock_Bayesian_BN(256, 384, stride=2, device=device)

        self.bayes_linear = BayesianSequential(
            FlattenLayer(384 * 32),
            BayesianLinear(384 * 32, 100, device=device),
            BayesianLeakyReLU(),
            BayesianLinear(100, num_class, device=device)
        )

    def forward(self, x):
        out, kl = self.bayes_conv0(x, 0)
        out = self.bn0(out)
        out, kl = self.leakyrelu0(out, kl)

        out, kl = self.bayes_layer1(out, kl)
        out, kl = self.bayes_layer2(out, kl)
        out, kl = self.bayes_layer3(out, kl)
        out, kl = self.bayes_layer4(out, kl)
        out, kl = self.bayes_layer5(out, kl)
        # print(out.shape)

        out, kl = self.bayes_linear(out, kl)

        return out, kl


if __name__ == '__main__':
    # one layer in Neural network
    layer = ResBlock_Bayesian_BN(33, 45, stride=1, device='cuda')
    temp = torch.randn(64, 33, 1000).cuda()
    tempkl = torch.tensor(0.0).cuda()
    temp_y, temp_kl = layer(temp, tempkl)
    print(temp_y.shape)

    # the entire Bayes Neural Network, with resent and Batch normalization
    model = Bayes_ResNet_BN(device='cuda', num_class=20)
    y, kl = model(temp)
    print(y.shape)

    # for training,
    model.mle = False
    beta = 1e-10
    crossentropy = None
    loss = crossentropy + beta * kl


    # for testing
    model.mle = True


