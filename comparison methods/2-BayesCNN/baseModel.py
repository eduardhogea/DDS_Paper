import torch
import torch.nn as nn
import torch.nn.functional as F
from distirbution import ScaleMixtureGaussian, NormalPosterior
from torch.nn.modules.utils import _pair
import math


class BayesianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self._mle = False
        # self._samples = samples

    @property
    def mle(self):
        return self._mle

    @mle.setter
    def mle(self, mle):
        for module in self._modules.values():
            if not isinstance(module, BayesianNN):
                raise AttributeError(f"{module} must be BayesianNN")
            module.mle = mle
        self._mle = mle

    def forward(self, x, kl):
        raise NotImplemented


# class BayesianModel(BayesianNN):
#     def __init__(self, samples):
#         super().__init__()
#         self._samples = samples
#         if self._mle:
#             self._samples = 1
#
#     @property
#     def samples(self):
#         return self._samples
#
#     @samples.setter
#     def samples(self, sample):
#         for module in self._modules.values():
#             if not isinstance(module, BayesianNN):
#                 raise AttributeError("Module must be BayesianNN")
#             module.samples = sample
#         self._samples = sample
#         if sample > 1:
#             self.mle = False
#
#     def forward_once(self, x):
#         raise NotImplementedError
#
#     def forward(self, x, kl):
#         batch_size = x.size(0)
#         outputs = torch.zeros(self.samples, batch_size, 10)
#         kls = torch.zeros(self.samples)
#         for i in range(self.samples):
#             outputs[i], kls[i] = self.forward_once(x)
#         return outputs.mean(0), kls.mean(0)


class BayesianSoftPlus(BayesianNN):    # softplus activation
    def forward(self, x, kl):
        return F.softplus(x), kl


class BayesianSequential(nn.Sequential, BayesianNN):
    def __init__(self, *args):
        super().__init__(*args)
        for module in self._modules.values():
            if not isinstance(module, BayesianNN):
                raise AttributeError(f"{module}  must be BayesianNN")

    def forward(self, x, kl):
        for module in self._modules.values():
            x, kl = module(x, kl)
        return x, kl


class BayesianMaxPool2d(nn.modules.pooling._MaxPoolNd, BayesianNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, kl):
        return F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices), kl


class BayesianMaxPool1d(nn.modules.pooling._MaxPoolNd, BayesianNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, kl):
        return F.max_pool1d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices), kl


class _BatchNormBayesian(BayesianNN):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, device, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, pi=.5, sigma1=math.exp(0), sigma2=math.exp(-6)):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.device = device
        self.prior = ScaleMixtureGaussian(pi=self.pi, sigma1=self.sigma1, sigma2=self.sigma2, device=device)

        if self.affine:
            self.weight_mean = nn.Parameter(torch.Tensor(num_features).to(device))
            self.weight_logvar = nn.Parameter(torch.Tensor(num_features).to(device))
            self.weight = NormalPosterior(self.weight_mean, self.weight_logvar, device)
            self.bias_mean = nn.Parameter(torch.Tensor(num_features).to(device))
            self.bias_logvar = nn.Parameter(torch.Tensor(num_features).to(device))
            self.bias = NormalPosterior(self.bias_mean, self.bias_logvar, device)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight_mean.data.uniform_(0, 1)
            self.weight_logvar.data = torch.ones_like(self.weight_logvar) * -4
            self.bias_mean.data.zero_()
            self.bias_logvar.data = torch.ones_like(self.weight_logvar) * -4
            # nn.init.uniform_(self.weight_mean)
            # nn.init.ones_(self.weight_logvar)
            # nn.init.zeros_(self.bias)

    def _check_input_dim(self, x):
        raise NotImplementedError

    def forward(self, x, kl):
        self._check_input_dim(x)
        if not self.mle:
            weight_sample = self.weight.sample()
            bias = self.bias.sample()
            log_var_post = self.weight.log_prob(weight_sample)
            log_prior = self.prior.log_prob(weight_sample)
            _kl = (log_var_post - log_prior).sum()
        else:
            weight_sample = self.weight.mu
            bias = self.bias.mu
            _kl = 0

        exponential_average_factor = torch.tensor(0.0).to(self.device)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = torch.tensor(self.momentum).to(self.device)
        print('fuck')
        return F.batch_norm(
            x, self.running_mean, self.running_var, weight_sample, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps), kl + _kl

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BayesianBatchNorm2D(_BatchNormBayesian):
    def __init__(self, num_features, device, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, device, eps, momentum, affine=affine,
                         track_running_stats=track_running_stats)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))


class BayesianBatchNorm1D(_BatchNormBayesian):
    def __init__(self, num_features, device, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, device, eps, momentum, affine=affine,
                         track_running_stats=track_running_stats)

    def _check_input_dim(self, x):
        if x.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(x.dim()))


class BatchNorm2DF(nn.BatchNorm2d, BayesianNN):
    def __init__(self,  num_features, device, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats)
        self.to(device)

    def forward(self, x, kl):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps), kl


class BatchNorm1DF(nn.BatchNorm1d, BayesianNN):
    def __init__(self, num_features, device, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats)
        self.to(device)

    def forward(self, x, kl):
        return F.batch_norm(x), kl


class BayesianReLU(BayesianNN):
    def __init__(self):
        super().__init__()

    def forward(self, x, kl):
        return F.relu(x), kl


class BayesianLeakyReLU(BayesianNN):
    def __init__(self):
        super().__init__()

    def forward(self, x, kl):
        return F.leaky_relu(x), kl


class FlattenLayer(BayesianNN):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x, kl):
        return x.view(-1, self.num_features), kl


class BayesianLinear(BayesianNN):
    def __init__(self, n_in, n_out, device, bias=True, pi=.5, sigma1=math.exp(0), sigma2=math.exp(-6)):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.pi = pi           # 0.5
        self.sigma1 = sigma1   # 1
        self.sigma2 = sigma2   # 0.002
        self.prior = ScaleMixtureGaussian(pi=pi, sigma1=sigma1, sigma2=sigma2, device=device)
        # =============变分推断用的proposal distribution：q（W）==================
        self.weight_mean = nn.Parameter(torch.Tensor(n_out, n_in).to(device))   # mean value of weight
        self.weight_logvar = nn.Parameter(torch.Tensor(n_out, n_in).to(device))  # log_variance of weight
        self.bias = bias
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(n_out, ).to(device))
            self.bias_logvar = nn.Parameter(torch.Tensor(n_out, ).to(device))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()    # initialization
        self.weight = NormalPosterior(self.weight_mean, self.weight_logvar, device)
        if bias:
            self.bias = NormalPosterior(self.bias_mean, self.bias_logvar, device)
        # =============================================================

    def reset_parameters(self):
        stv = 10 / (self.n_in ** .5)
        self.weight_mean.data.uniform_(-stv, stv)       # initial value of weight
        self.weight_logvar.data.uniform_(-stv, stv).add_(-4)
        if self.bias is not None:
            self.bias_mean.data.uniform_(-stv, stv)
            self.bias_logvar.data.uniform_(-stv, stv).add_(-4)

    def fcprobforward(self, x, kl):
        if not self.mle:    # mle is a flag
            # If mle = False, for training process
            weight_sample = self.weight.sample()    # sampling ：mu + sigma * eps
            if self.bias:
                bias = self.bias.sample()
            else:
                bias = None
            out = F.linear(x, weight=weight_sample, bias=bias)

            # KL -- ELBO
            log_var_post = self.weight.log_prob(weight_sample)    #
            log_prior = self.prior.log_prob(weight_sample)
            _kl = (log_var_post - log_prior).sum()
            return out, kl + _kl
        else:
            # if mle = True，for testing
            weight = self.weight.mu
            if self.bias:
                bias = self.bias.mu
            else:
                bias = None
            out = F.linear(x, weight=weight, bias=bias)

            # KL
            log_var_post = self.weight.log_prob(weight)
            log_prior = self.prior.log_prob(weight)
            _kl = (log_var_post - log_prior).sum()
            return out, kl + _kl

    def forward(self, x, kl):
        return self.fcprobforward(x, kl)

    def extra_repr(self):
        return 'in={n_in}, out={n_out}, mle={_mle}'.format(**self.__dict__)


class _BayesianConvnd(BayesianNN):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, output_padding, groups, bias, pi, sigma1, sigma2, device):
        super(_BayesianConvnd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.pi = pi
        # self.samples = samples
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.conv_mean = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).to(device))
        # print(self.conv_mean.shape)
        self.conv_logvar = nn.Parameter(torch.Tensor(out_channels,
                                                     in_channels // groups, *kernel_size).to(device))
        self.prior = ScaleMixtureGaussian(pi=self.pi, sigma1=self.sigma1, sigma2=self.sigma2, device=device)
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_channels).normal_(0, .1))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.conv = NormalPosterior(self.conv_mean, self.conv_logvar, device)
        if bias:
            self.bias = NormalPosterior(self.bias_mean, self.bias_logvar, device)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stv = 1. / math.sqrt(n)
        self.conv_mean.data.uniform_(-stv, stv)
        self.conv_logvar.data.uniform_(-stv, stv).add_(-4)
        if self.bias is not None:
            self.bias_mean.data.uniform_(-stv, stv)
            self.bias_logvar.data.uniform_(-stv, stv).add_(-4)

    def forward(self, x, kl):
        raise NotImplementedError

    def extra_repr(self):
        return '{in_channels}, {out_channels},' \
               'kernel_size={kernel_size}, stride={stride}, padding={padding},' \
               ' mle={_mle}'.format(**self.__dict__)


class BayesianConv2d(_BayesianConvnd):
    def __init__(self, in_channels, out_channels, kernel_size, device,
                 stride=1, padding=0, dilation=1, groups=1, bias=None,
                 pi=.5, sigma1=math.exp(0), sigma2=math.exp(-6)):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, _pair(0), groups, bias, pi, sigma1, sigma2, device)

    def convprobforward(self, x, kl):
        if self.mle:
            weight_sample = self.conv.mu
        else:
            weight_sample = self.conv.sample()
        if self.bias:
            if self.mle:
                bias = self.bias.mu
            else:
                bias = self.bias.sample()
        else:
            bias = None
        out = F.conv2d(x, weight_sample, bias, self.stride, self.padding, self.dilation, self.groups)

        # KL
        log_var_post = self.conv.log_prob(weight_sample)
        log_prior = self.prior.log_prob(weight_sample)
        _kl = (log_var_post - log_prior).sum()
        return out, kl + _kl

    def forward(self, x, kl):
        return self.convprobforward(x, kl)

class BayesianConv1d(BayesianNN):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=None, pi=.5,
                 sigma1=math.exp(0), sigma2=math.exp(-6)):

        super(BayesianConv1d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.pi = pi
        # self.samples = samples
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.conv_mean = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).to(device))
        # print(self.conv_mean.shape)
        self.conv_logvar = nn.Parameter(torch.Tensor(out_channels,
                                                     in_channels // groups, kernel_size).to(device))
        self.prior = ScaleMixtureGaussian(pi=self.pi, sigma1=self.sigma1, sigma2=self.sigma2, device=device)
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_channels).normal_(0, .1))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.conv = NormalPosterior(self.conv_mean, self.conv_logvar, device)
        if bias:
            self.bias = NormalPosterior(self.bias_mean, self.bias_logvar, device)

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size

        stv = 1. / math.sqrt(n)
        self.conv_mean.data.uniform_(-stv, stv)
        self.conv_logvar.data.uniform_(-stv, stv).add_(-4)
        if self.bias is not None:
            self.bias_mean.data.uniform_(-stv, stv)
            self.bias_logvar.data.uniform_(-stv, stv).add_(-4)

    def convprobforward(self, x, kl):
        if self.mle:
            weight_sample = self.conv.mu
        else:
            weight_sample = self.conv.sample()
        if self.bias:
            if self.mle:
                bias = self.bias.mu
            else:
                bias = self.bias.sample()
        else:
            bias = None
        # print(weight_sample.shape)
        out = F.conv1d(x, weight_sample, bias, self.stride, self.padding, self.dilation, self.groups)

        # KL
        log_var_post = self.conv.log_prob(weight_sample)
        log_prior = self.prior.log_prob(weight_sample)
        _kl = (log_var_post - log_prior).sum()
        return out, kl + _kl

    def forward(self, x, kl):
        return self.convprobforward(x, kl)

    def extra_repr(self):
        return '{in_channels}, {out_channels},' \
               'kernel_size={kernel_size}, stride={stride}, padding={padding},' \
               ' mle={_mle}'.format(**self.__dict__)


class GaussianVariationalInference(nn.Module):
    def __init__(self, loss=nn.MSELoss()):
        super(GaussianVariationalInference, self).__init__()
        self.loss = loss

    def forward(self, logits, y, kl, beta):
        loss = self.loss(logits, y) + beta*kl
        return loss