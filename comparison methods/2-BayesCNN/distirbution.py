import torch
from torch.distributions import Normal
import math


class Posterior:

    def sample(self):
        raise NotImplementedError

    def log_prob(self, inp):
        raise NotImplementedError


class NormalPosterior(Posterior):
    # 后验概率，对数似然函数
    def __init__(self, mu, rho, device):
        self.mu = mu.to(device)    # 均值
        self.rho = rho.to(device)   # 对数方差
        self.normal = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    @property
    def sigma(self):
        # self.rho.exp() = e的rho次方
        # torch.log1p = ln( 1 + e的rho次方 )
        # 维度是[n_out, n_in]
        return torch.log1p(self.rho.exp())   # 为什么要用log1p而不是log，并没有算多准呀
    # 难道是因为用变分所给的proposal distribution是N{mu, log[1+exp(rho)]}

    def sample(self):
        eps = self.normal.sample(self.rho.shape).squeeze(-1)   # 维度[n_out, n_in]
        # 重参数技巧：eps服从标准正太分布，从而使得sigma可导
        return self.mu + self.sigma * eps

    def log_prob(self, inp):
        # 高斯分布的对数似然函数，将求积变为求和
        res = (-math.log(2 * math.pi) / 2 - self.sigma.log() - ((inp - self.mu) ** 2) / (2 * (self.sigma ** 2))).sum()
        # self.sigma.log() => 对sigma的每个元素求自然对数
        return res


class Prior:

    def log_prob(self, inp):
        raise NotImplementedError


class ScaleMixtureGaussian(Prior):
    # 先验概率，对数似然函数
    def __init__(self, pi, sigma1, sigma2, device):
        # sigma2设的很小的，是为了让sigma1所代表的分布能变小吗（相当于是加了一个限制？）
        self.pi = pi
        if isinstance(sigma1, float):
            self.sigma1 = torch.Tensor([sigma1]).to(device)
        else:
            self.sigma1 = sigma1.to(device)
        if isinstance(sigma2, float):
            self.sigma2 = torch.Tensor([sigma2]).to(device)
        else:
            self.sigma2 = sigma2.to(device)
        # self.sigma2 = sigma2.to(device)
        self.normal1 = Normal(0, sigma1)
        print(self.normal1)
        self.normal2 = Normal(0, sigma2)

    def log_prob(self, inp):
        log_prob1 = self.normal1.log_prob(inp)   # 计算inp在定义的正太分布normal1中对应的概率的对数
        # 维度是[n_out, n_In]
        # 对inp中的每一个元素，带入高斯分布的对数概率密度函数中，
        # 例如，inp的第一个元素是x，normal1有均值、方差，
        # 计算得到：-（x-均值）**2 / 2*方差**2 - ln（方差）- 0.5*ln（2*3.1415）
        log_prob2 = self.normal2.log_prob(inp)

        # log_prob1.exp()还原真实概率， pi是平衡两个分布的权重
        # 对数似然函数
        res = torch.log(1e-16 + self.pi * log_prob1.exp() + (1 - self.pi) * log_prob2.exp()).sum()
        # 该值越大越好，由实验可知：随着inp的减小，res会变大
        # 我明白了，一开始的时候inp是在N（0，1）分布采样，到了后面，这个正则项会逼着inp在N（0，0.0025）分布采样
        # 就是控制了权重Weight的方差会变小
        return res


class FixedNormal(Prior):
    # takes mu and logvar as float values and assumes they are shared across all weights
    def __init__(self, mu, logvar, device):
        self.mu = mu.to(device)
        self.logvar = logvar.to(device)
        super(FixedNormal, self).__init__()

    def log_prob(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * math.exp(self.logvar))


if __name__ == '__main__':
    pr = ScaleMixtureGaussian(pi=1, sigma1=math.exp(-2), sigma2=math.exp(-6), device="cpu")
    pr2 = FixedNormal(torch.tensor(0), torch.tensor(-3), "cpu")
    data = torch.randn(10)
    print(pr.log_prob(data))
    print(pr2.log_prob(data).sum())
    # n = NormalPosterior(torch.zeros(2000, 3),  torch.ones(2000, 3))
    # m = n.sample()
    # print(m.std(0), m.mean(0))