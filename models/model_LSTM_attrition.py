# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

"""
Neural Networks model : LSTM_Attrition
"""


class LSTM(nn.Module):

    def __init__(self, seq_length, hidden_size, dropout=0.75, num_layer=2):
        super(LSTM, self).__init__()

        # lstm
        self.lstm = nn.LSTM(seq_length, hidden_size, dropout=dropout, num_layers=num_layer)

        # linear
        self.hidden2label = nn.Conv1d(48, 48, 7, stride=1)

        # weight
        self.weight = nn.Conv1d(48, 48, 1, stride=1, bias=False)

        # dropout
        self.module_name = "LSTM_Attrition"

    def forward(self, x):
        # lstm
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        logit = self.hidden2label(lstm_out)
        logit = self.weight(logit)
        logit = logit.squeeze()
        return logit


def get_LSTM(seq_length, hidden_size, num_layer=3):
    return LSTM(seq_length, hidden_size, num_layer=num_layer)


if __name__ == '__main__':
    x = torch.randn(7, 32, 48)
    net = LSTM(48, 48)
    out = net(x)
    print(out.shape)