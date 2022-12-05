"""
Author: https://github.com/FrancescoSaverioZuppichini/ResNet
"""


import numpy as np
import sys
import torch
import torch.nn as nn

import torch.nn.functional as F

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
            'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                               stride=self.downsampling, bias=False),
            'bn': nn.BatchNorm2d(self.expanded_channels)

        })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels,
                    self.conv, kernel_size=1),
            activation(),
            conv_bn(self.out_channels, self.out_channels, self.conv,
                    kernel_size=3, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels,
                    self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBottleNeckBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **
                  kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet50(nn.Module):

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[3, 4, 6, 3],
                                    activation=nn.ReLU, block=ResNetBottleNeckBlock, encode=True, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.layer2 = ResNetLayer(64, 64, n=deepths[0], activation=activation)
        self.layer3 = ResNetLayer(
            64*4, 128, n=deepths[0], activation=activation)
        self.layer4 = ResNetLayer(
            128*4, 256, n=deepths[0], activation=activation)
        self.layer5 = ResNetLayer(
            256*4, 512, n=deepths[0], activation=activation)

        self.encode = encode

    def forward(self, x):
        l1 = self.layer1(x.float())
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        if self.encode:
            return l5, [l1, l2, l3, l4, l5]
        else:
            return l5


if __name__ == '__main__':
    img = torch.rand((4, 3, 256, 512))
    # encoder_50 = ResNetEncoder(3,block=ResNetBottleNeckBlock,deepths=[3,4,6,3])
    # output = encoder_50(img)
    # print(output.shape)
    encoder = ResNet50(3, deepths=[3,4,6,3],encode=False)
    output = encoder(img)
    print(output.shape)
