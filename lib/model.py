#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:55:50 2021

@author: ayman
"""
import torch
import torch.nn as nn
import torchvision as tv



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

        self.conv1 = nn.Conv2d(in_channels=shape[0], out_channels=32, kernel_size=8, stride=4, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=False)
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




class ResNetDDQN(nn.Module):
    """
    Pre-trained ResNet 18 with the following features:
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

        self.conv1 = nn.Conv2d(in_channels=shape[0], out_channels=32, kernel_size=8, stride=4, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        linear_input_ = self.frwdConv(torch.zeros(1, *shape)).shape[1]

        self.val1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.vact1 = nn.ReLU()
        self.val2 = nn.Linear(in_features=256, out_features=1)

        self.pol1 = nn.Linear(in_features=linear_input_, out_features=256)
        self.pact1 = nn.ReLU()
        self.pol2 = nn.Linear(in_features=256, out_features=actions)

    def frwdConv(self, o):
        fo = o.float()/255
        y = self.act1(self.norm1(self.conv1(fo)))
        y = self.act2(self.norm2(self.conv2(y)))
        y = self.act3(self.norm3(self.conv3(y)))
        return torch.flatten(y, start_dim=1)

    def frwdLin(self, y):
        val = self.vact1(self.val1(y))
        val = self.val2(val)
        pol = self.pact1(self.pol1(y))
        pol = self.pol2(pol)
        return val, pol

    def forward(self, x):
        fx = self.frwdConv(x)
        val, prob = self.frwdLin(fx)
        return prob, val