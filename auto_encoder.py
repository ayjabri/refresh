#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder in pytorch
Simple linear and convolutional examples on MNIST 
The network takes in MNIST photos and reconstructs them

Created on Thu Jun 17 13:46:04 2021

@author: ayman
"""

import os
import torch
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt

PATH = '/home/ayman/workspace/RNN/dataset/mnist/'


class LinEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(features, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     )

        self.decoder = nn.Sequential(nn.Linear(128, features),
                                     nn.Tanh(),
                                     nn.ReLU(),
                                     )

    def forward(self, x):
        encode = self.encoder(x)
        return self.decoder(encode)


class ConvEncoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.encoder = nn.Sequential(nn.Conv2d(shape[0], 64, 5, 2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 96, 3, 2),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(),
                                     )

        self.decoder = nn.Sequential(nn.ConvTranspose2d(96, 64, 5, 1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64, 32, 5, 1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(
                                         32, shape[0], 6, 2, padding=1),
                                     nn.Tanh(),
                                     nn.ReLU())

    def forward(self, x):
        x = x.view(-1, *self.shape)
        y = self.encoder(x)
        return self.decoder(y).flatten(1)


@torch.no_grad()
def evaluate(net, test_loader, device='cpu', show=False):
    correct = 0
    net.device = device
    net.to(device)
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        output = net(img).softmax(dim=1).argmax(dim=1)
        correct += (output == label).sum().item()
        del img, label
    accuracy = correct/len(test_loader.dataset)
    if show:
        print(f"Accuracy: {accuracy:.2}%")
    return accuracy


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 2000
    features = 28*28
    num_classes = 10
    learning_rate = 1e-3
    device = 'cuda'
    num_epochs = 20

    # Prepare data
    if not os.path.exists(PATH):
        os.makedirs(os.path.join(PATH, 'train'))
        os.makedirs(os.path.join(PATH, 'test'))

    # tfms = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize(mean=(0.485,), std=(0.229,), inplace=True)])
    tfms = tv.transforms.ToTensor()
    # train_set = tv.datasets.MNIST(root=PATH+'train/', train=True,
    #                               transform=tfms, download=True)
    # test_set = tv.datasets.MNIST(root=PATH+'test/', train=False,
    #                              transform=tfms, download=True)
    
    train_set = tv.datasets.CIFAR10(root=PATH+'train/', train=True,
                                  transform=tfms, download=True)
    test_set = tv.datasets.CIFAR10(root=PATH+'test/', train=False,
                                 transform=tfms, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=4, drop_last=False)
    
    img,label = next(iter(train_set))
    shape = img.shape
    net = ConvEncoder(shape).to(device)
    
    # evaluate(net, test_loader, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fun = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for img, label in train_loader:
            img = img.view(-1, features).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = net(img)
            loss_v = loss_fun(output, img)
            loss_v.backward()
            optimizer.step()
            total_loss += loss_v.item()
        # accuracy = evaluate(net, test_loader, device)
        # , Testing Accuracy: {accuracy:.2f}%')
        print(f'Epoch:{epoch:3}, Loss:{total_loss:7.3f}')
