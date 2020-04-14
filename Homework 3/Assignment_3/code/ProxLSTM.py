
import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    def __init__(self,lstm, epsilon=1.0):	# feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm   # use LSTMCell as blackbox
        self.epsilon = epsilon



    def forward(self, input, pre_h, pre_c):

        # input [batch, input_size]
        '''need to be implemented'''
        vt = input
        ht, st = self.lstm(vt, (pre_h,pre_c))
        st.requires_grad = True
        import pdb; pdb.set_trace()
        
         with torch.enable_grad():
            st = ag.Variable(st, requires_grad=True)
            vt = ag.Variable(vt, requires_grad=True)
            gt = ag.grad(st[:,0], vt, retain_graph=True)[0]
        
        gg = torch.mm(gt,gt.T)
        ct = torch.mm(torch.inverse(torch.eye(gt.size(0)) + self.epsilon*gg), st)
        
        return (ht, ct)



                        dL/dh  dL/dc
    def backward(self, grad_h, grad_c):
        
        
        dL/dc

        return d ht/ d input, d ct/d input
    
        return dL/dst, dL/dgt

        '''need to be implemented'''


