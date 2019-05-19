# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class AttentionModel(nn.Module):
    def __init__(self, batch_size, seq_length, hidden_size, dropout=0.75, num_layer=2):
        super(AttentionModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        hidden_sie : Size of the hidden_state of the LSTM

        --------

        """
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(seq_length, hidden_size, dropout=dropout, num_layers=num_layer)
        self.label = nn.Linear(hidden_size, 1)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, x, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits


if __name__ == '__main__':
    x = torch.randn(1, 32, 46)
    net = AttentionModel(32, 46, 50)
    out = net(x)
    print(out.shape)