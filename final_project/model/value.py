import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ValueNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(outplanes - 1, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes - 1)
        x = F.relu(self.linear1(x))
        score = torch.tanh(self.linear2(x))
        return score