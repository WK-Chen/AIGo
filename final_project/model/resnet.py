import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import *


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shorcut = nn.Sequential()

        # make sure the shortcut should have the same dimension
        if stride != 1 or in_planes != self.expansion * planes:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shorcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, inplanes, outplanes):
        """
        inplanes: Shape of the input state
        INPLANES = (HISTORY + 1) * 2 + 1
        outplanes: Probabilities for all moves
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        for block in range(BLOCKS):
            setattr(self, "res{}".format(block), Block(outplanes, outplanes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS):
            out = getattr(self, "res{}".format(block))(out)

        out = getattr(self, "res{}".format(BLOCKS - 1))(out)
        return out
