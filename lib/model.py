#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:55:50 2021

@author: ayman
"""
import torch
import torch.nn as nn
import torchvision as tv
# import numpy as np


class DDQN(nn.Module):
    """
    Duel DeepQ Neural Network.

    shape: Tuple of (N Stack, Height, Width)
    actions: Discrete number of action from action_space
    """

    def __init__(self, shape, actions):
        super().__init__()
        self.shape = shape
        self.actions = actions

        self.conv1 = nn.Conv2d(
            in_channels=shape[0], out_channels=32, kernel_size=8, stride=4, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        linear_input_ = self._frwdConv(torch.zeros(1, *shape)).shape[1]

        self.val1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.vact1 = nn.ReLU()
        self.val2 = nn.Linear(in_features=256, out_features=1)

        self.adv1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.aact1 = nn.ReLU()
        self.adv2 = nn.Linear(in_features=256, out_features=actions)

    def _frwdConv(self, o):
        fo = o.float()/255
        y = self.act1(self.norm1(self.conv1(fo)))
        y = self.act2(self.norm2(self.conv2(y)))
        y = self.act3(self.norm3(self.conv3(y)))
        return torch.flatten(y, start_dim=1)

    def _frwdLin(self, y):
        val = self.vact1(self.val1(y))
        val = self.val2(val)
        adv = self.aact1(self.adv1(y))
        adv = self.adv2(adv)
        return val, adv

    def forward(self, x):
        """Feed forward a batch of float tensors."""
        fx = self._frwdConv(x)
        val, adv = self._frwdLin(fx)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class ResNetDDQN(nn.Module):
    """
    Pre-trained ResNet 18 with the following features.

    1- Fix the first 3 Convu
    ** Experemental
    """

    def __init__(self, shape, actions):
        super().__init__()
        assert shape[0] == 3

        self.shape = shape
        self.actions = actions
        self._load_resnet()

        conv_out_shape = self.resnet(torch.zeros(1, *shape)).shape[1]

        self.val = nn.Sequential(nn.Linear(conv_out_shape, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.adv = nn.Sequential(nn.Linear(conv_out_shape, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, actions))

    def _load_resnet(self):
        freez = ['layer1', 'layer2', 'layer3']
        self.resnet = tv.models.resnet18(True)
        for l in freez:
            for p in getattr(self.resnet, l).parameters():
                p.requires_grad = False
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        """Feed forward a batch of float tensors."""
        fx = x.float()/255
        rx = self.resnet(fx)
        val = self.val(rx)
        adv = self.adv(rx)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class A2CNet(nn.Module):
    """
    Actor-Critic network with two heads.

    shape: Tuple of (N Stack, Height, Width)
    probs: distribution over actions
    vals: estimation of the state value
    """

    def __init__(self, shape, actions):
        super().__init__()
        self.shape = shape
        self.actions = actions

        self.conv1 = nn.Conv2d(
            in_channels=shape[0], out_channels=32, kernel_size=8, stride=4, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        linear_input_ = self._frwdConv(torch.zeros(1, *shape)).shape[1]

        self.val1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.vact1 = nn.ReLU()
        self.val2 = nn.Linear(in_features=256, out_features=1)

        self.pol1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.pact1 = nn.ReLU()
        self.pol2 = nn.Linear(in_features=256, out_features=actions)

    def _frwdConv(self, o):
        fo = o.float()/255
        y = self.act1(self.norm1(self.conv1(fo)))
        y = self.act2(self.norm2(self.conv2(y)))
        y = self.act3(self.norm3(self.conv3(y)))
        return torch.flatten(y, start_dim=1)

    def _frwdLin(self, y):
        val = self.vact1(self.val1(y))
        val = self.val2(val)
        pol = self.pact1(self.pol1(y))
        pol = self.pol2(pol)
        return val, pol

    def forward(self, x):
        """Feed forward a batch of float tensors."""
        fx = self._frwdConv(x)
        val, prob = self._frwdLin(fx)
        return prob, val


class A2Cgru(nn.Module):
    """
    Actor-Critic network with two heads.

    shape: Tuple of (N Stack, Height, Width)
    probs: distribution over actions
    vals: estimation of the state value
    """

    def __init__(self, shape, actions, hidden_size=1024):
        super().__init__()
        self.shape = shape
        self.actions = actions

        self.conv1 = nn.Conv2d(
            in_channels=shape[0], out_channels=32, kernel_size=8, stride=4, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        conv_output_ = self._frwdConv(torch.zeros(1, *shape)).shape[1]

        self.gru = nn.GRUCell(conv_output_, hidden_size)
        self.gact = nn.ReLU()
        self.val1 = nn.Linear(in_features=hidden_size, out_features=256)
        self.vact1 = nn.ReLU()
        self.val2 = nn.Linear(in_features=256, out_features=1)

        self.pol1 = nn.Linear(in_features=conv_output_, out_features=256)
        self.pact1 = nn.ReLU()
        self.pol2 = nn.Linear(in_features=256, out_features=actions)

    def _frwdConv(self, o):
        fo = o.float()/255
        y = self.act1(self.norm1(self.conv1(fo)))
        y = self.act2(self.norm2(self.conv2(y)))
        y = self.act3(self.norm3(self.conv3(y)))
        return torch.flatten(y, start_dim=1)

    def _frwdLin(self, y):
        val = self.gru(y)
        val = self.gact(val)
        val = self.vact1(self.val1(val))
        val = self.val2(val)
        pol = self.pact1(self.pol1(y))
        pol = self.pol2(pol)
        return val, pol

    def forward(self, x):
        """Feed forward a batch of float tensors."""
        fx = self._frwdConv(x)
        val, prob = self._frwdLin(fx)
        return prob, val


class ConvBN(nn.Module):
    # convolutional layer then Batchnorm
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


class DarknetBlock(nn.Module):
    # The basic blocs.
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class Darknet(nn.Module):
    # Replicates the table 1 from the YOLOv3 paper
    def __init__(self, num_blocks, num_classes=1000):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_group_layer(32, num_blocks[0])
        self.layer2 = self.make_group_layer(64, num_blocks[1], stride=2)
        self.layer3 = self.make_group_layer(128, num_blocks[2], stride=2)
        self.layer4 = self.make_group_layer(256, num_blocks[3], stride=2)
        self.layer5 = self.make_group_layer(512, num_blocks[4], stride=2)
        self.linear = nn.Linear(1024, num_classes)

    def make_group_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in, ch_in*2, stride=stride)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in*2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return F.log_softmax(self.linear(out))
