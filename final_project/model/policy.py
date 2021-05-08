import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(PolicyNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsofmax = nn.LogSoftmax(dim=1)
        self.linear = nn.Linear(outplanes - 1, outplanes)

    def foward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = self.linear(x)
        prob = self.logsofmax(x).exp()
        return prob