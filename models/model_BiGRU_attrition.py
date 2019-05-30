# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_BiGRU.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


"""
Neural Networks model : Bidirection GRU
"""


class BiGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout=0.75, num_layer=2):
        super(BiGRU, self).__init__()
        # gru
        self.bigru = nn.GRU(input_size, hidden_size, dropout=dropout, num_layers=num_layer, bidirectional=True)

        self.fc = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        # linear
        self.hidden2label = nn.Linear(hidden_size * 2, hidden_size)
        # bn
        self.bn = nn.BatchNorm1d(hidden_size*2)
        # weight
        self.weight = nn.Conv1d(hidden_size * 2, hidden_size * 2, 7, bias=False)

        #  dropout
        self.dropout = nn.Dropout(dropout)
        self.module_name = "BiGRU_attrition"

    def forward(self, input):
        # embed = self.dropout(input)
        # input = embed.view(len(input), embed.size(1), -1)
        # gru
        gru_out, _ = self.bigru(input)
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.tanh(gru_out)
        # gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = self.fc(gru_out)
        gru_out = self.bn(gru_out)
        gru_out = self.weight(gru_out).squeeze()
        # gru_out = torch.tanh(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        logit = y
        return logit


def get_BiGRU_attrition(seq_length, hidden_size, num_layer=2):
    return BiGRU(seq_length, hidden_size, num_layer=num_layer)


if __name__ == '__main__':
    x = torch.randn(7, 32, 48)
    net = BiGRU(48, 48)
    out = net(x)
    print(out.shape)