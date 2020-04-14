import torch
import torch.nn as nn
import numpy as np
from conv import Conv


class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size, m=14):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()
        
        # crf param
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.m = m
        
        # conv layer params
        self.out_channels = 1 # output channel of conv layer
        self.conv_layers = conv_layers
        self.stride = (1,1)
        self.padding = True
        self.cout_shape = self.get_cout_dim() # output shape of conv layer
        self.cout_numel = self.cout_shape[0]*self.cout_shape[1]
        
        
        
        self.init_params()

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        self.conv = Conv(self.conv_layers[0][-1], self.out_channels, padding=self.padding,stride=self.stride)
        self.W = torch.randn(self.num_labels, self.cout_numel, requires_grad=True)
        self.T = torch.randn(self.num_labels, self.num_labels, requires_grad=True)
    
    def get_cout_dim(self):
        if self.padding:
            return (int(np.ceil(self.input_dim[0]/self.stride[0])), int(np.ceil(int(self.input_dim[1]/self.stride[1]))))
        return None
    
    
    # X: (batch_size, 14, 16, 8) dimensional tensor
    # iterates over all words in a batch, and decodes them one by one
    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """        
        decods = torch.zeros(self.batch_size, self.m, 1, dtype=torch.int)
        for i in range(self.batch_size):
            # Reshape the word to (14,1,16,8)
            word = X[i].reshape(self.m, 1, self.input_dim[0],self.input_dim[1])
            # conv operation performed for one word independently to every letter
            features = self.get_conv_features(word)
            # now decode the sequence using conv features
            decods[i] = self.dp_infer(features)

        return decods
    
    # input: x: (m, d), m is # of letters a word has, d is the feature dimension of letter image
    # input: w: (26, d), letter weight vector
    # input: T: (26, 26), letter-letter transition matrix
    # output: letter_indices: (m, 1), letter labels of a word
            
    # decode a sequence of letters for one word
    def dp_infer(self, x):
        w = self.W
        T = self.T
        m = self.m
    
        pos_letter_value_table = torch.zeros((m, 26), dtype=torch.float64)
        pos_best_prevletter_table = torch.zeros((m, 26), dtype=torch.int)
        # for the position 1 (1st letter), special handling
        # because only w and x dot product is covered and transition is not considered.
        for i in range(26):
        # print(w)
        # print(x)
            pos_letter_value_table[0, i] = torch.dot(w[i, :], x[0, :])
        
        # pos_best_prevletter_table first row is all zero as there is no previous letter for the first letter
        
        # start from 2nd position
        for pos in range(1, m):
        # go over all possible letters
            for letter_ind in range(self.num_labels):
                # get the previous letter scores
                prev_letter_scores = pos_letter_value_table[pos-1, :].clone()
                # we need to calculate scores of combining the current letter and all previous letters
                # no need to calculate the dot product because dot product only covers current letter and position
        	        # which means it is independent of all previous letters
                for prev_letter_ind in range(self.num_labels):
                    prev_letter_scores[prev_letter_ind] += T[prev_letter_ind, letter_ind]
        
                # find out which previous letter achieved the largest score by now
                best_letter_ind = torch.argmax(prev_letter_scores)
                # update the score of current positive with current letter
                pos_letter_value_table[pos, letter_ind] = prev_letter_scores[best_letter_ind] + torch.dot(w[letter_ind,:], x[pos, :])
                # save the best previous letter for following tracking to generate most possible word
                pos_best_prevletter_table[pos, letter_ind] = best_letter_ind
        letter_indicies = torch.zeros((m, 1), dtype=torch.int)
        letter_indicies[m-1, 0] = torch.argmax(pos_letter_value_table[m-1, :])
        max_obj_val = pos_letter_value_table[m-1, letter_indicies[m-1, 0]]
        # print(max_obj_val)
        for pos in range(m-2, -1, -1):
            letter_indicies[pos, 0] = pos_best_prevletter_table[pos+1, letter_indicies[pos+1, 0]]
        return letter_indicies


    def loss(self, X, labels):
        
        
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        features = self.get_conv_features(X)
        loss = blah
        return loss

    def backward(self):
        
        
        """
        Return the gradient of the CRF layer
        :return:
        """
        gradient = blah
        return gradient


    # performs conv operation to every (16,8) image in the word. m = 14 (default) - word length
    # returns flattened vector of new conv features
    def get_conv_features(self, word):
        """
        Generate convolution features for a given word
        """
        cout = self.conv.forward(word)
        cout = cout.reshape(cout.shape[0], self.cout_numel)
        return cout








