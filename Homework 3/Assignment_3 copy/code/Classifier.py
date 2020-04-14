
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import ProxLSTM as pro



class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, input_size):
		super(LSTMClassifier, self).__init__()
		

		self.output_size = output_size	# should be 9
		self.hidden_size = hidden_size  #the dimension of the LSTM output layer
		self.input_size = input_size	  # should be 12
		self.normalize = F.normalize()
		self.conv = nn.Conv1d(in_channels= self.input_size, out_channels= 64, kernel_size= 3, stride= 3) # feel free to change out_channels, kernel_size, stride
		self.relu = nn.ReLU()
		self.lstm = nn.LSTMCell(64, hidden_size)
		self.linear = nn.Linear(self.hidden_size, self.output_size)


		
	def forward(self, input, r, batch_size, mode='plain'):
		# do the forward pass
		# pay attention to the order of input dimension.
		# input now is of dimension: batch_size * sequence_length * input_size


		'''need to be implemented'''
		if mode == 'plain'
				# chain up the layers

		if mode == 'AdvLSTM'
				# chain up the layers
			  # different from mode='plain', you need to add r to the forward pass
			  # also make sure that the chain allows computing the gradient with respect to the input of LSTM

		if mode == 'ProxLSTM'
				# chain up layers, but use ProximalLSTMCell here
