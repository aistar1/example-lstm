from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from torch.utils.data import  Dataset, DataLoader

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, batch_first=True):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, hidden_cell = self.lstm(input_seq)
        predictions = self.linear(lstm_out)
        return predictions[:, -1]

if __name__ == '__main__':

    # https://www.cnblogs.com/xiximayou/p/15036715.html

    # torch.Size([64, 32, 300])：表示[batchsize, max_length, embedding_size]
    # hidden_size = 128
    # output_size = 1
    # lstm = nn.LSTM(300, 128, batch_first=True, num_layers=1)
    # linear = nn.Linear(hidden_size, output_size)
    # output, (hn, cn) = lstm(torch.rand(64, 32, 300))
    # predictions = linear(output)
    
    # print(output.shape)
    # print(hn.shape)
    # print(cn.shape)
    # print(predictions.shape)
    # print(predictions[:,-1].shape)



    model = LSTM()
    print(model)
    y_pred = model(torch.rand(64, 32, 1)) # (batch, seq, feature)
    print(y_pred.shape)
