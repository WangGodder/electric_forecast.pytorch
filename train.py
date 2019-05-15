# -*- coding: utf-8 -*-
# @Time    : 2019/5/16
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.optim import *
from models import *
from dataset import *

criterion = nn.MSELoss()
epochs = 100
batch_size = 64
learning_rate = 10 ** (-1.0)
use_gpu = True

seq_length = 46
data_length = 1
hidden_size = 100

def train():
    net = get_BiGRU(seq_length, hidden_size)
    if use_gpu:
        net = net.cuda()
    optimizer = Adam(net.parameters(), lr=learning_rate)
    dataset = get_dataset(seq_length, data_length)
    dataloader = DataLoader(dataset, batch_size, num_workers=8)
    for epoch in range(epochs):
        dataset.train()
        net.train()
        for (input, label) in tqdm(dataloader):
            # transpose input and label from shape (N, data length, -1) to (data length, N, -1)
            input = torch.transpose(input, 0, 1)
            label = torch.transpose(label, 0, 1)
            if use_gpu:
                input = input.cuda()
                label = label.cuda()

            out = net(input)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dataset.valid()
        avg_loss = valid(net, dataloader)
        print("epoch: %3d, avg loss in valid: %4.3f" % (epoch, avg_loss))


def valid(net: nn.Module, dataloader: DataLoader):
    net.eval()
    sum_loss = []
    for (input, label) in tqdm(dataloader):
        # transpose input and label from shape (N, data length, -1) to (data length, N, -1)
        input = torch.transpose(input, 0, 1)
        label = torch.transpose(label, 0, 1)

        if use_gpu:
            input = input.cuda()
            label = label.cuda()
            net = net.cuda()

        out = net(input)
        loss = criterion(out, label)
        sum_loss.append(loss.data)
    return sum(sum_loss) / len(sum_loss)