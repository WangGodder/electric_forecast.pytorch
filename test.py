# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder

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


seq_length = 48
data_length = 7
hidden_size = 48
layer_num = 4


pth = "./checkpoints/LSTM_Attrition.pth"


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


def test(input):
    net = get_BiLSTM_attrition(seq_length, data_length, layer_num)
    net.load_state_dict(pth)

    out = net(input)
    return out


if __name__ == '__main__':
    dataset = get_dataset(48, 7, False)
    input = dataset.all_data[766 : 763, 0: 48]
    label = dataset.all_data[764, 0: 48]

    out = test(input)
    for ind in range(1, 49):
        plotter.plot("predict", "real", ind, label)