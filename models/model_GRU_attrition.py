# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_GRU.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


"""
Neural Networks model : GRU
"""


class GRU(nn.Module):
    
    def __init__(self, seq_length, hidden_size, dropout=0.75, num_layer=2):
        super(GRU, self).__init__()
        # gru
        self.gru = nn.GRU(seq_length, hidden_size, dropout=dropout, num_layers=num_layer)
        self.weight = nn.Conv1d(1, 1, hidden_size, stride=hidden_size, bias=False)
        # linear
        self.hidden2label = nn.Linear(hidden_size, hidden_size)
        #bn
        self.bn = nn.BatchNorm1d(hidden_size)
        # weight
        #  dropout
        self.dropout = nn.Dropout(dropout)
        self.module_name = 'GRU_attrition'

    def forward(self, input):
        lstm_out, _ = self.gru(input)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        # lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = self.bn(lstm_out)
        lstm_out = self.weight(lstm_out).squeeze()
        # linear
        y = self.hidden2label(lstm_out)
        logit = y
        return logit


def get_GRU_attrition(seq_length, hidden_size, num_layer=2):
    return GRU(seq_length, hidden_size, num_layer=num_layer)


if __name__ == '__main__':
    x = torch.randn(1, 32, 48 * 7)
    net = GRU(48 * 7, 48 * 7)
    out = net(x)
    print(out.shape)