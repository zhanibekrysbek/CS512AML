
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import ProxLSTM as pro

torch.manual_seed(1)

class LSTMClassifier(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, input_size, prox_eps = 0.1):
        super(LSTMClassifier, self).__init__()

        self.history = [[],[]] # trainACC, testACC
        
        self.output_size = output_size	# should be 9
        self.hidden_size = hidden_size  # the dimension of the LSTM output layer
        self.input_size = input_size	  # should be 12
        self.normalize = F.normalize
        self.oChannels = 64
        self.prox_eps = prox_eps

        # feel free to change out_channels, kernel_size, stride
        self.conv = nn.Conv1d(in_channels= self.input_size, 
                        out_channels= self.oChannels, kernel_size= 3, stride=1) 
        self.relu = nn.ReLU()
        self.lstm = nn.LSTMCell(self.oChannels, self.hidden_size)
        #self.lstm = nn.LSTM(self.oChannels, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        



    def forward(self, inX, r, batch_size, mode='plain'):
        # do the forward pass
        # pay attention to the order of input dimension.
        # input now is of dimension: batch_size * sequence_length * input_size


        '''need to be implemented'''
        if mode == 'plain':
            # chain up the layers

            inX = self.normalize(inX, dim=2)
            inX = inX.permute(0, 2, 1) # [batch, input, seq]
            inX = self.conv(inX)
            inX = self.relu(inX)
            newSeqLen = inX.shape[2]
            
            """
            # nn.LSTM implementation
            inX = inX.permute(2,0,1) # [seq, batch, input]
            self.inX = inX
            lstmOut, _ = self.lstm(inX)
            lstmOut = lstmOut[-1,:,:]
            #lstmOut = self.linear(lstmOut[-1,:,:])
            """
            
            # nn.LSTMCELL implementation:
            
            inX = inX.permute(2, 0, 1) # [seq, batch, input]
            hx = torch.zeros(batch_size, self.hidden_size)/100.0
            cx = torch.zeros(batch_size, self.hidden_size)/100.0
            self.inX = inX
            for i in range(newSeqLen):
                hx, cx = self.lstm(inX[i], (hx, cx))
                lstmOut = hx
            
            out = self.linear(lstmOut)
            #out = self.relu(out)
        
        if mode == 'AdvLSTM':
            # chain up the layers
            # nt from mode='plain', you need to add r to the forward pass
            # also make sure that the chain allows computing the gradient with respect to the input of LSTM
            
            inX = self.normalize(inX, dim=2)
            inX = inX.permute(0, 2, 1) # [batch, input, seq]
            inX = self.conv(inX)
            inX = self.relu(inX)
            newSeqLen = inX.shape[2]
            """
            # nn.LSTM implementation
            inX = inX.permute(2,0,1) # [seq, batch, input]
            self.inX = inX
            inX += r
            lstmOut, _ = self.lstm(inX)
            lstmOut = lstmOut[-1,:,:]
            #lstmOut = self.linear(lstmOut[-1,:,:])
            """
            
            # nn.LSTMCELL implementation:
            inX = inX.permute(2, 0, 1) # [seq, batch, input]
            self.inX = inX
            inX += r
            hx = torch.zeros(batch_size, self.hidden_size)/100.0
            cx = torch.zeros(batch_size, self.hidden_size)/100.0
            for i in range(newSeqLen):
                hx, cx = self.lstm(inX[i], (hx, cx))
                lstmOut = hx
            
            out = self.linear(lstmOut)
        
        if mode == 'ProxLSTM':
            
            self.ProxLSTM = pro.ProximalLSTMCell(self.lstm, epsilon = self.prox_eps)
            # layers, but use ProximalLSTMCell here
            inX = self.normalize(inX, dim=2)
            inX = inX.permute(0, 2, 1) # [batch, input, seq]
            inX = self.conv(inX)
            inX = self.relu(inX)
            newSeqLen = inX.shape[2]
            
            """
            # nn.LSTM implementation
            inX = inX.permute(2,0,1) # [seq, batch, input]
            self.inX = inX
            inX += r
            lstmOut, _ = self.lstm(inX)
            lstmOut = lstmOut[-1,:,:]
            #lstmOut = self.linear(lstmOut[-1,:,:])

            """
            # nn.LSTMCELL implementation:
            inX = inX.permute(2, 0, 1) # [seq, batch, input]
            self.inX = inX
            inX += r
            hx = torch.randn(batch_size, self.hidden_size)/100.0
            cx = torch.randn(batch_size, self.hidden_size)/100.0
            for i in range(newSeqLen):
                hx, cx = self.ProxLSTM(inX[i], hx, cx)
                lstmOut = hx
            
            out = self.linear(lstmOut)
            
        return out
    
    

    
    
