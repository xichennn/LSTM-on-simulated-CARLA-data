#add a MLP to encode traffic state
import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from tqdm import trange
import random

class CarlaNet(nn.Module):
    def __init__(self, args) -> None:
        super(CarlaNet, self).__init__()

        ## unpack args
        self.args = args
        self.input_feature = args["input_size"]
        self.output_feature = args["output_size"]
        self.embedding_size = args["embedding_size"]
        # self.dyn_embedding_size = args['dyn_embedding_size']
        self.hidden_size_enc = args["hidden_size_enc"]
        self.hidden_size_dec = args["hidden_size_dec"]
        self.num_layers = args['num_layers']
        self.target_len = args['target_len']
        self.ts_feature = args["ts_feature"]
        self.ts_embedding_size = args["ts_embedding_size"]

        ## define network weights

        # input embeddding layer
        self.inp_emb = nn.Linear(self.input_feature, self.embedding_size)

        # Encoder LSTM
        self.enc_lstm = nn.LSTM(self.embedding_size, self.hidden_size_enc, self.num_layers)

        # traffic state embedding 
        self.ts_emb = nn.Linear(self.ts_feature, self.ts_embedding_size)

        # Decoder LSTM
        self.dec_lstm = nn.LSTM(self.ts_embedding_size+self.hidden_size_enc, self.hidden_size_dec, self.num_layers)

        # output layers
        self.oup = nn.Linear(self.hidden_size_dec, self.output_feature)

        #Activations
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self,hist,ts):

        #forward pass hist
        hist_embedded = self.relu(self.inp_emb(hist))
        _,(hist_enc,_) = self.enc_lstm(hist_embedded)
        # hist_enc = self.leaky_relu(self.dyn_emb(hist_enc))
        ts_enc = self.relu(self.ts_emb(ts))

        #concatenate encodings
        enc = torch.cat((hist_enc, ts_enc), 2)

        #decode
        fut_pred = self.decode(enc)

        return fut_pred

    def decode(self, enc):
        enc = enc.repeat(self.target_len, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        fut_pred = self.oup(h_dec)
        return fut_pred

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
       
        # self.training_prediction = args['training_prediction'] 

        self.num_layers = args['num_layers']

        ## Define network weights
        self.linear1 = nn.Linear(self.input_size, self.embedding_size)
        self.lstm1 = nn.LSTM(self.embedding_size, self.hidden_size_enc, self.num_layers)
        # self.lstm_decoder = nn.LSTM(self.hidden_size_enc, self.hidden_size_dec, self.num_layers)
        # #output layer
        # self.linear = nn.Linear(self.hidden_size_dec, self.input_size)
        
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
        # hist_enc = hist_enc.view(hist_enc.shape[1], hist_enc.shape[2])
        # #decoder pass
        # predictions = self.decode(hist_enc)

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
        self.output_size = args["input_size"]
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
        # hist_dec = hist_dec.view(hist_dec.shape[1], hist_dec.shape[2])
        # #decoder pass
        # predictions = self.decode(hist_enc)

        #use hidden state to predict
        outputs = self.linear2(hist_dec[0])

        return outputs, hist_dec
