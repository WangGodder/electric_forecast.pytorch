# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_BiLSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""
Neural Networks model : Bidirection LSTM attrition
"""


class BiLSTM(nn.Module):
    
    def __init__(self, seq_length, hidden_size, dropout=0.75, num_layer=2):
        super(BiLSTM, self).__init__()

        self.bilstm = nn.LSTM(seq_length, hidden_size // 2, num_layers=num_layer, dropout=dropout, bidirectional=True, bias=False)

        self.fc = nn.Conv1d(hidden_size, hidden_size, 7, stride=1)
        self.hidden2label1 = nn.Conv1d(hidden_size, hidden_size // 2, 1)
        self.hidden2label2 = nn.Conv1d(hidden_size // 2, hidden_size, 1)
        #weight
        self.weight = nn.Conv1d(hidden_size, hidden_size, 1, bias=False)
        # self.dropout = nn.Dropout(config.dropout)
        self.module_name = 'BiLSTM'

    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        # bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = self.fc(bilstm_out)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)

        logit = y
        logit = self.weight(logit)
        return logit


def get_BiLSTM(seq_length, hidden_size, num_layer=2):
    return BiLSTM(seq_length, hidden_size, num_layer=num_layer)


if __name__ == '__main__':
    x = torch.randn(7, 32, 48)
    net = BiLSTM(48, 48)
    out = net(x)
    print(out.shape)