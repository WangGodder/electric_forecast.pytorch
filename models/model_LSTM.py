# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_LSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init

"""
Neural Networks model : LSTM
"""


class LSTM(nn.Module):
    
    def __init__(self, seq_length, hidden_size, dropout=0.75, num_layer=2):
        super(LSTM, self).__init__()

        # lstm
        self.lstm = nn.LSTM(seq_length, hidden_size, dropout=dropout, num_layers=num_layer)

        # if args.init_weight:
        #     print("Initing W .......")
        #     # n = self.lstm.input_size * self.lstm
        #     init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))

        # linear
        self.hidden2label = nn.Linear(hidden_size, 1)
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # lstm
        lstm_out, _ = self.lstm(x)
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        # lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # lstm_out = F.tanh(lstm_out)
        # linear
        logit = self.hidden2label(lstm_out)
        return logit


def get_LSTM(seq_length, hidden_size, num_layer=1):
    return LSTM(seq_length, hidden_size, num_layer=num_layer)


if __name__ == '__main__':
    x = torch.randn(46, 32, 1)
    net = LSTM(1, 50)
    out = net(x)
    print(out.shape)