
import torch
import torch.nn as nn
import torch.autograd as ag
from scipy import optimize
from scipy.optimize import check_grad
import numpy

class ProximalLSTMCell(ag.Function):
    def __init__(self,lstm, epsilon=1.0, batch_size=27):	# feel free to add more input arguments as needed
        super(ProximalLSTMCell, self).__init__()
        self.lstm = lstm   # use LSTMCell as blackbox
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.hidden_size = self.lstm.hidden_size
        self.input_size = self.lstm.input_size



    def forward(self, vt, pre_h, pre_c):

        # vt [batch, input_size]
        '''need to be implemented'''
        Gt = torch.zeros(self.batch_size, self.hidden_size, self.input_size)
        with torch.enable_grad():
            vt = ag.Variable(vt, requires_grad=True)
            ht, st = self.lstm(vt, (pre_h,pre_c))
            for i in range(st.size(-1)):
                gt = ag.grad(st[:,i], vt, grad_outputs=torch.ones_like(st[:,0]), retain_graph=True)[0]
                Gt[:,i,:] = gt
        
        GtT = Gt.transpose(1,2)
        gg = torch.matmul(Gt,GtT)
        I = torch.eye(gg.size(1))
        I = I.reshape((1, gg.size(1),gg.size(1)))
        I = I.repeat(self.batch_size, 1, 1)
        
        st= st.reshape((self.batch_size, gg.size(1),1))
        ct = torch.matmul(torch.inverse(I + self.epsilon*gg), st)
        ct = ct.reshape((self.batch_size, gg.size(1)))
        return (ht, ct)



#                        dL/dh  dL/dc
    def backward(self, grad_h, grad_c):
        
        '''
        dL/dc

        return d ht/ d input, d ct/d input
    
        return dL/dst, dL/dgt
        '''
        
        '''need to be implemented'''


