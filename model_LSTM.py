import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from tqdm import trange
import random


class lstm_encoder(nn.Module):
    def __init__(self, args) -> None:
        '''
        Arguments:
        input_size:         number of features in input X
        embedding size:     embedding size
        hidden_size:        number of features in the hidden state h
        num_layers:         number of recurrent layers (e.g. 2 means two stacked LSTMs)

        '''
        super(lstm_encoder, self).__init__()
        self.args = args
        self.input_size = args["input_size"]
        self.hidden_size_enc = args["hidden_size_enc"]
        self.embedding_size = args["embedding_size"]

        self.num_layers = args['num_layers']

        ## Define network weights
        self.linear1 = nn.Linear(self.input_size, self.embedding_size)
        self.lstm1 = nn.LSTM(self.embedding_size, self.hidden_size_enc, self.num_layers)
        
        # #activation functions
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, hist, hidden):
        '''
        Arguments:
        hist:        input to the network, shape (seq_length, batch_size, #features)
        hidden:      initial hidden state

        Returns:     
        hidden:      final hidden state
        '''
        #encoder pass
        embedded = self.relu(self.linear1(hist))

        #pass both hidden state and cell memory in hist_enc
        _, hist_enc= self.lstm1(embedded, hidden)

        return hist_enc

    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : batch_size:          hist.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size_enc),
                torch.zeros(self.num_layers, batch_size, self.hidden_size_enc))

class lstm_decoder(nn.Module):
    def __init__(self, args) -> None:
        '''
        Arguments:
        embedding_size:     Embedding size
        hidden_size:        Hidden size of LSTM
        output_size:        number of features in the output
        '''

        super(lstm_decoder, self).__init__()
        self.args = args
        self.embedding_size = args["embedding_size"]
        self.hidden_size_dec = args["hidden_size_dec"]
        self.input_size = args["input_size"]
        self.output_size = args["output_size"]
        self.num_layers = args['num_layers']
        
        self.linear1 = nn.Linear(self.output_size, self.embedding_size)
        self.lstm1 = nn.LSTM(self.embedding_size, self.hidden_size_dec, self.num_layers)
        self.linear2 = nn.Linear(self.hidden_size_dec, self.output_size)
       
        self.relu = nn.ReLU()
    
    def forward(self, hist, hidden):
        '''
        Arguments:
        hist:        input to the network, shape (seq_length, batch_size, #features)
        hidden:      initial hidden state

        Returns:  
        outputs:      output from lstm   
        hidden:      final hidden state
        '''
        #encoder pass
        embedded = self.relu(self.linear1(hist))
        _, hist_dec = self.lstm1(embedded, hidden)

        #use hidden state to predict
        outputs = self.linear2(hist_dec[0])

        return outputs, hist_dec

