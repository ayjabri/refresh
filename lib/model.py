#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:55:50 2021

@author: ayman
"""
import torch
import torch.nn as nn


class DQNetConv(nn.Module):
    def __init__(self, shape, actions, device='cpu'):
        super().__init__()
        self.shape = shape
        self.features = shape[0]
        self.actions = actions
        self.device = device

        self.input = nn.Conv2d(in_channels=self.features,
                               out_channels=32, kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)
        in_features = self.__getConvSize__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=actions)

        self.to(device)

    def __frwdConv__(self, x):
        y = torch.relu(self.input(x))
        y = torch.relu(self.conv1(y))
        return torch.relu(self.conv2(y))

    def __frwdFc__(self, x):
        y = torch.relu(self.fc1(x))
        return self.output(y)

    def __getConvSize__(self):
        x = torch.zeros(1, *self.shape)
        o = self.__frwdConv__(x)
        o = torch.flatten(o, start_dim=1)
        return o.shape[1]

    def forward(self, x):
        fx = x.float()/255
        y = self.__frwdConv__(fx)
        y = torch.flatten(y, start_dim=1)
        y = self.__frwdFc__(y)
        return y


class DuelDQNet(nn.Module):
    def __init__(self, shape, actions):
        super().__init__()
        self.shape = shape
        self.actions = actions

        self.conv = nn.Sequential(nn.Conv2d(shape[0], 32, 8, 4),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 2, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Flatten())

        conv_out_shape = self.conv(torch.zeros(1, *shape)).shape[1]

        self.val = nn.Sequential(nn.Linear(conv_out_shape, 256),
                                 # nn.LayerNorm(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.adv = nn.Sequential(nn.Linear(conv_out_shape, 256),
                                 # nn.LayerNorm(256),
                                 nn.ReLU(),
                                 nn.Linear(256, actions))

    def forward(self, x):
        fx = x.float()/255
        y = self.conv(fx)
        val = self.val(y)
        adv = self.adv(y)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DDQN(nn.Module):
    def __init__(self, shape, actions):
        super().__init__()
        self.shape = shape
        self.actions = actions

        self.conv1 = nn.Conv2d(
            in_channels=shape[0], out_channels=32, kernel_size=8, stride=4)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        linear_input_ = self.frwdConv(torch.zeros(1, *shape)).shape[1]
        self.val1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.vact1 = nn.ReLU()
        self.val2 = nn.Linear(in_features=256, out_features=1)

        self.adv1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.aact1 = nn.ReLU()
        self.adv2 = nn.Linear(in_features=256, out_features=actions)

    def frwdConv(self, o):
        fo = o.float()/255
        y = self.act1(self.norm1(self.conv1(fo)))
        y = self.act2(self.norm2(self.conv2(y)))
        y = self.act3(self.norm3(self.conv3(y)))
        return torch.flatten(y, start_dim=1)

    def frwdLin(self, y):
        val = self.vact1(self.val1(y))
        val = self.val2(val)
        adv = self.aact1(self.adv1(y))
        adv = self.adv2(adv)
        return val, adv

    def forward(self, x):
        fx = self.frwdConv(x)
        val, adv = self.frwdLin(fx)
        return val + (adv - adv.mean(dim=1, keepdim=True))
