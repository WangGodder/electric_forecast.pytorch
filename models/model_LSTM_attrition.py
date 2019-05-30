# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from opt import opt
"""
Neural Networks model : LSTM_Attrition
"""


class LSTM(nn.Module):

    def __init__(self, seq_length, hidden_size, dropout=0.75, num_layer=2):
        super(LSTM, self).__init__()

        # lstm
        self.lstm = nn.LSTM(seq_length, hidden_size, dropout=dropout, num_layers=num_layer)

        self.lstm2 = nn.LSTM(hidden_size, opt.hidden_size2, dropout=dropout, num_layers=num_layer)

        # weight
        self.weight = nn.Conv1d(opt.hidden_size2, opt.hidden_size2, 1, stride=1, bias=False)

        # linear
        self.hidden2label = nn.Conv1d(opt.hidden_size2, opt.fc, 7, stride=1)

        # softmax
        self.classify = nn.Softmax(dim=1)

        # dropout
        self.module_name = "LSTM_Attrition"

        # self._init_weight()

    def forward(self, x):
        # lstm
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = self.lstm2(lstm_out)
        # lstm_out, _ = self.lstm3(lstm_out)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        logit = self.weight(lstm_out)
        logit = F.relu(logit, inplace=True)

        logit = self.hidden2label(logit)
        logit = logit.squeeze()
        # logit = F.relu(logit, inplace=True)
        logit = self.classify(logit)
        return logit

    def _init_weight(self):
        nn.init.constant_(self.weight.weight, 2 ** -8.)


def get_LSTM_attrition(seq_length, hidden_size, num_layer=3):
    return LSTM(seq_length, hidden_size, num_layer=num_layer)


if __name__ == '__main__':
    x = torch.randn(7, 32, 48)
    net = LSTM(48, 48)
    out = net(x)
    print(out.shape)