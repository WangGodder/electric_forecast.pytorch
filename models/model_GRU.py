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
        # linear
        self.hidden2label = nn.Linear(hidden_size, 1)
        #  dropout
        self.dropout = nn.Dropout(dropout)
        self.module_name = 'GRU'

    def forward(self, input):
        lstm_out, _ = self.gru(input)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # linear
        y = self.hidden2label(lstm_out)
        logit = y
        return logit


def get_GRU(seq_length, hidden_size, num_layer=1):
    return GRU(seq_length, hidden_size, num_layer=num_layer)