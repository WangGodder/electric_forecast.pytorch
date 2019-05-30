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
from dataset import get_dataset
import sys
import datetime
from util import Logger
import os
import numpy as np
from visdom import Visdom
from opt import opt


criterion = nn.MSELoss()
epochs = 500
batch_size = 32
use_gpu = True

seq_length = 48
data_length = 7
hidden_size = opt.hidden_size1
layer_num = 4


t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
sys.stdout = Logger(os.path.join('.', 'logs', t + 'layer_num' + str(layer_num) + '.txt'))


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='challenge'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')


plotter = VisdomLinePlotter(env_name='electric forecast')


def train():
    learning_rate1 = 10 ** (-1.0)
    learning_rate2 = 10 ** (-1.0)
    learning_rate1 = np.linspace(learning_rate1, np.power(10, -6.), epochs + 1)
    learning_rate2 = np.linspace(learning_rate2, np.power(10, -6.), epochs + 1)
    lr1 = learning_rate1[0]
    lr2 = learning_rate2[0]
    # net = get_BiGRU(seq_length, hidden_size)
    lstm = get_LSTM(seq_length, hidden_size, layer_num)
    bi_lstm = get_BiLSTM(seq_length, hidden_size, layer_num)
    lstm_att = get_LSTM_attrition(seq_length, hidden_size, layer_num)
    bi_lstm_att = get_BiLSTM_attrition(seq_length, hidden_size, layer_num)
    gru = get_GRU(seq_length, hidden_size, layer_num)
    bi_gru = get_BiGRU(seq_length, hidden_size, layer_num)
    gru_att = get_GRU_attrition(seq_length, hidden_size, layer_num)
    bi_gru_att = get_BiGRU_attrition(seq_length, hidden_size, layer_num)

    if use_gpu:
        lstm = lstm.cuda()
        bi_lstm = bi_lstm.cuda()
        lstm_att = lstm_att.cuda()
        bi_lstm_att = bi_lstm_att.cuda()
        gru = gru.cuda()
        bi_gru = bi_gru.cuda()
        gru_att = gru_att.cuda()
        bi_gru_att = bi_gru_att.cuda()

    nets = [lstm, bi_lstm, lstm_att, bi_lstm_att]

    # optimizer = Adam(net.parameters(), lr=lr)
    opt_lstm = Adam(lstm.parameters(), lr=lr1)
    opt_bi_lstm = Adam(bi_lstm.parameters(), lr=lr2)
    opt_lstm_att = Adam(lstm_att.parameters(), lr=lr1)
    opt_bi_lstm_att = Adam(bi_lstm_att.parameters(), lr=lr2)
    opt_gru = Adam(gru.parameters(), lr=lr1)
    opt_bi_gru = Adam(bi_gru.parameters(), lr=lr2)
    opt_gru_att = Adam(gru_att.parameters(), lr=lr1)
    opt_bi_gru_att = Adam(bi_gru_att.parameters(), lr=lr2)

    opts = [opt_lstm, opt_bi_lstm, opt_lstm_att, opt_bi_lstm_att]

    dataset = get_dataset(seq_length, data_length, True)
    dataloader = DataLoader(dataset, batch_size, num_workers=8)
    max_mape = [1, 1, 1, 1]
    for epoch in range(epochs):
        dataset.train()
        # net.train()
        for ind in range(len(nets)):
            net = nets[ind]
            net.train()
            optimizer = opts[ind]
            for data, label in tqdm(dataloader):
                # transpose input and label from shape (N, data length, -1) to (data length, N, -1)
                data = torch.transpose(data, 0, 1)
                label = torch.transpose(label, 0, 1)
                label = label.squeeze()
                if use_gpu:
                    data = data.cuda()
                    label = label.cuda()

                out = net(data)

                loss = criterion(out, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dataset.valid()
        # avg_loss, mape = valid(lstm, dataloader)
        # print("epoch: %3d, avg loss in valid: %4.3f, mape: %3.3f" % (epoch, avg_loss, mape))
        to_valid(nets, dataloader, epoch, max_mape)
        lr1 = learning_rate1[epoch + 1]
        lr2 = learning_rate2[epoch + 1]
    print("best mape: ")
    for ind in range(len(nets)):
        print("%s : %3.3f" % (nets[ind].module_name, max_mape[ind]))


def to_valid(nets: list, dataloader, epoch, max_mape):
    for index, net in enumerate(nets):
        avg_loss, mape = valid(net, dataloader)
        print("epoch: %3d, %s avg loss in valid: %4.3f, mape: %3.3f" % (epoch, net.module_name, avg_loss, mape))
        module_name = str(net.module_name)
        plotter.plot('loss', module_name, epoch, avg_loss.item())
        plotter.plot('mape', module_name, epoch, mape.item())
        if mape < max_mape[index]:
            max_mape[index] = mape
            torch.save(net, "checkpoints/" + net.module_name + ".pth")


def valid(net: nn.Module, dataloader: DataLoader):
    net.eval()
    sum_loss = []
    sum_mape = []
    for (data, label) in tqdm(dataloader):
        # transpose input and label from shape (N, data length, -1) to (data length, N, -1)
        data = torch.transpose(data, 0, 1)
        label = torch.transpose(label, 0, 1)
        label = label.squeeze()

        if use_gpu:
            data = data.cuda()
            label = label.cuda()
            net = net.cuda()

        out = net(data)
        loss = criterion(out, label)
        sum_loss.append(loss.data)
        mape = calc_mape(label, out)
        sum_mape.append(mape.data)
    return sum(sum_loss) / len(sum_loss), sum(sum_mape) / len(sum_loss)


def calc_mape(label: torch.Tensor, out: torch.Tensor):
    loss = label - out
    mape = torch.mean(torch.div(torch.abs(loss), torch.abs(label)))
    return mape


if __name__ == '__main__':
    train()