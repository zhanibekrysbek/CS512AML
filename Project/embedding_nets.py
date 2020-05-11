import torch
import torchvision
import torchvision.datasets as tdata
import torchvision.transforms as tTrans
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optm
import matplotlib.pyplot as plt
# Python Imaging Library
import PIL
import numpy as np
import sys as sys
import utils
from utils import Print

#



# --------------------------------------------------------------------------------------------------------

# Model Definition
class MNISTNet(nn.Module):
    """ DESCRIPTION : This is a classic MNIST classificationj net based on LeNet. It is used for demonstration purposes
                      on how models should be declared (not of the hierarchical type such as siamese or VAE, or autoencoder).  
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self):
        super(MNISTNET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'MNIST_NET'
        self.classMethod = 'standard'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Output here is of 24x24 dimnesions
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # output of conv2 is of 20x20
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
   
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # output here is 10 x 12x12
        x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        # output here is 20 x 4x4 = 320 params
        # Flatten in to 1D to feed to dense Layer.
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    # ------------------

    def forward_no_drop(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        # forward return format should be: predictions and then any other misc stuff the used loss might need such
        # as temperature etc.
        return x

    # ------------------
    def predict(self, x):
        return self.forward(x)
    # ------------------

    def report(self):

        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================

# =========================================================================================================================================================================================
# FRUITS NETWORKS
# =========================================================================================================================================================================================

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AGADAKOS NET    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ANET(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                  dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':50}, 'linear2':{'out':120}},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(ANET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'A_NET'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h1, w1 = utils.comp_conv_dimensions(dataChannels1, dataSize[1],dataSize[2], dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = utils.comp_pool_dimensions(dataChannels1, h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = utils.comp_conv_dimensions(dataChannels1, h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = utils.comp_pool_dimensions(dataChannels1, h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, dims['linear1']['out'])
        self.fc2 = nn.Linear(dims['linear1']['out'], dims['linear2']['out'])
        
        # Device handling CPU or GPU 
        self.device = device
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.mPool1(x))
        x = self.conv2(x)
        x = F.relu(self.mPool2(x))
        x = F.dropout(x, training=self.training)
        #print(x.shape)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)        
        return x
    
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MIRZA NET    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
class MNET(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                  dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':50}, 'linear2':{'out':120}},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(ANET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'A_NET'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h1, w1 = utils.comp_conv_dimensions(dataChannels1, dataSize[1],dataSize[2], dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = utils.comp_pool_dimensions(dataChannels1, h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = utils.comp_conv_dimensions(dataChannels1, h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = utils.comp_pool_dimensions(dataChannels1, h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, dims['linear1']['out'])
        self.fc2 = nn.Linear(dims['linear1']['out'], dims['linear2']['out'])
        
        # Device handling CPU or GPU 
        self.device = device
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.mPool1(x))
        x = self.conv2(x)
        x = F.relu(self.mPool2(x))
        x = F.dropout(x, training=self.training)
        #print(x.shape)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)        
        return x
    
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RYSBEK NET    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RNET(nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                  dims = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':50}, 'linear2':{'out':120}},
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(ANET, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'A_NET'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.classMethod = 'label'
        self.trainMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h1, w1 = utils.comp_conv_dimensions(dataChannels1, dataSize[1],dataSize[2], dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        print("Dims conv1 for linear layaer: {} {}".format(h1,w1))
        # Compute dims after 1nd max layer
        h2, w2 = utils.comp_pool_dimensions(dataChannels1, h1, w1, dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        print("Dims pool1 for linear layaer: {} {}".format(h2,w2))
        # Compute dims after 2nd  layer
        h3, w3 = utils.comp_conv_dimensions(dataChannels1, h2,w2, dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        print("Dims conv2 for linear layaer: {} {}".format(h3,w3))
        # Compute dims after 2nd max layer
        h4, w4 = utils.comp_pool_dimensions(dataChannels1, h3,w3, dims['pool2']['kSize'], stride = dims['pool2']['stride'])
        print("Dims pool2 for linear layaer: {} {}".format(h4,w4))
        self.linearSize = dims['conv2']['nodes'] * h4*w4
        print("Dims for linear layaer: " + str(self.linearSize))
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.drop = nn.Dropout2d()
        self.mPool1 = torch.nn.MaxPool2d(dims['pool1']['kSize'], stride = dims['pool1']['stride'])
        self.mPool2 = torch.nn.MaxPool2d(dims['pool2']['kSize'], stride = dims['pool2']['stride']) 
        self.fc1 = nn.Linear(self.linearSize, dims['linear1']['out'])
        self.fc2 = nn.Linear(dims['linear1']['out'], dims['linear2']['out'])
        
        # Device handling CPU or GPU 
        self.device = device
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.mPool1(x))
        x = self.conv2(x)
        x = F.relu(self.mPool2(x))
        x = F.dropout(x, training=self.training)
        #print(x.shape)
        x = x.view(-1, self.linearSize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)        
        return x
    
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------

    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

# ===============================================================================================================================================================================
# ENCODER NETS
# ===============================================================================================================================================================================

class BasicEncoder (nn.Module):
    """ DESCRIPTION : THis is one of the 3 classifier netoworks designed for the ffruits dataset!
    """
    # Class variables for measures.
    accuracy = 0
    trainLoss= 0
    testLoss = 0
    def __init__(self, device = 'cpu', dataSize = [3,100,100],
                 dims2 = {'conv1':{'nodes':10,'kSize':3, 'stride':1 }, 'conv2':{'nodes':10,'kSize':3, 'stride':1}, 
                          'pool1':{'stride':(1,1), 'kSize':(2,1)}, 'pool2':{'stride':(1,1), 'kSize':(2,1)},
                          'linear1':{'out':200}},
                 dims = {'nodes':[32,64,64,64], 'kSizes':[3,3,3,3], 'strides':[2,2,2,2], 'latentDim':200},
                 decDims = {'nodes':[64,64,64,32], 'kSizes':[3,3,3,3], 'strides':[2,2,2,2], 'latentDim':200},
                 decoder = None, verbose = False,
                 **kwargs):
        """ DESCRIPTIONS: Used to initialzie the model. This is the workhorse. pass all required arguments in the init
                          as the example here, hwere the device (cpu or gpu), the datasize and the dims for the various layers
            are required to be pass. The arguments can be anything and it is also a good idea to **kwargs in the end, to be able to handle extra 
            arguments if need be, in the future; better to pass extra arguments and ignore them than get errors!
        
        """
        super(BasicEncoder, self).__init__()
        
        # Boiler plate code. Any init should declare the following
        self.descr = 'Basic_Encoder'
        self.lr_policy = 'plateau'
        self.metric = 0
        self.trainMethod = 'label'
        self.classMethod = 'label'
        self.propLoss = nn.CrossEntropyLoss()
    # ======================================================================
    # Section A.0
    # ***********
        # Handle input arguments here
        # ********
        self.numOfConvLayers = len(dims['nodes'])
        self.latentDim = dims['latentDim']
        self.verbose = verbose
        self.device = device
        # Layer Dimensions computation
        # Compute  layer dims after 1nd conv layer, automatically.
        dataChannels1 = '2d' # 3d for 3 dimensional data or 2d. Used to correctly comptue the output dimensions of conv and pool layers!
        h, w = [], []
        h.append(dataSize[1])
        w.append(dataSize[2])
        for i in range(1,self.numOfConvLayers+1):
            hi, wi = utils.comp_conv_dimensions(dataChannels1, h[i-1], w[i-1], dims['kSizes'][i-1], stride = dims['strides'][i-1])
            h.append(hi)
            w.append(wi)
            print("Dims conv1 for linear layaer: {} {}".format(hi,wi))
        
        self.linearSize = dims['nodes'][-1] * h[-1]*w[-1]
        print("Dims for linear layaer: " + str(self.linearSize))
        self.h, self.w = h,w
        self.latentDim = dims['latentDim']
        # ---|
        
    # Section A.1
    # ***********
        # Skeleton of this network; the blocks to be used.
        # Similar to Fischer prize building blocks!
        
        # Layers Declaration
        self.convLayers  = [nn.Conv2d(dataSize[0], dims['nodes'][0], kernel_size=dims['kSizes'][0], stride = dims['strides'][0])]
        self.convLayers += [nn.Conv2d(dims['nodes'][i-1], dims['nodes'][i], kernel_size=dims['kSizes'][i], stride = dims['strides'][i]) for i in range(1,self.numOfConvLayers)]
        self.convLayers = nn.ModuleList(self.convLayers)
        #self.conv1 = nn.Conv2d(dataSize[0], dims['conv1']['nodes'], kernel_size=dims['conv1']['kSize'], stride = dims['conv1']['stride'])
        #self.conv2 = nn.Conv2d(dims['conv1']['nodes'], dims['conv2']['nodes'], kernel_size=dims['conv2']['kSize'], stride = dims['conv2']['stride'])
        self.toLatentSpace = nn.Linear(self.linearSize, dims['latentDim'])
        
        # Device handling CPU or GPU 
        self.device = device
        
        if decoder is None:
            self.decoder = nn.Sequential(
                Print(self.verbose,div=True),
                nn.Linear(self.latentDim, self.linearSize),
                nn.ReLU(),
                Print(enable=self.verbose,div=True),
                utils.shapedUnFlatten(dims['nodes'][-1], self.h[-1], self.w[-1]),
                Print(enable=self.verbose,div=True),
                nn.ConvTranspose2d(dims['nodes'][-1], dims['nodes'][-2], kernel_size=dims['kSizes'][-1], stride=dims['strides'][-1], output_padding=0),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-2], dims['nodes'][-3], kernel_size=dims['kSizes'][-2], stride=dims['strides'][-2], output_padding=1),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-3], dims['nodes'][-4], kernel_size=dims['kSizes'][-3], stride=dims['strides'][-3], output_padding=0),
                nn.ReLU(),
                Print(enable=self.verbose),
                nn.ConvTranspose2d(dims['nodes'][-4], dataSize[0], kernel_size=dims['kSizes'][-4], stride=dims['strides'][-4], output_padding=1),
                nn.Sigmoid(),
                Print(enable=self.verbose),
            )
        else:
            self.decoder = decoder
    # Init End
    # --------|
    
    # SECTION A.2
    # ***********
    # Set the aove defined building blocks as an
    # organized, meaningful architecture here.
    def forward(self, x):
        encodedX = self.encode(x)
        decodedX = self.decode(encodedX)
        return [x, decodedX]
    # ------------------
    def encode(self,x):
        for i in range(len(self.convLayers)):
            x = F.relu(self.convLayers[i](x))
            #print(x.shape)
        x = x.reshape(-1, self.linearSize)
        x = self.toLatentSpace(x)
        return x
    # ------------------
    def decode(self, x):
        decX = self.decoder(x) 
        return decX
    # ------------------
    def visualize():
        pass
    # ------------------
    def predict(self, x, **kwargs):
        return self.forward(x)
    # ------------------
    def generate(self, sample=None, bSize=32,  **kwargs):
        """ DESCRIPTION: This function will generate an image based on an input ssample, usually encoded from the forward sample.
                         Sample can be a list of samples, where their average will be used to generate the final image. 
            ARGUMENTS: sample (list of tensors): Hold the samples to be used to generate new fruits from, can be any number
                                                 and it should be in ether in [batachSize,channels, h,w] or(channels,h,w) format .
                                                 If NONE, function will generate new data from sampling the latent space.
                       bSize (int): Batch size then the function generates data by sampling from latent space only. IS NOT USED when
                                    generating from input images.
            RETURNS: genData (tensor): Returned tensor shame shape as the original input, the decoded output that is either:
                                       a) The average represnetation of the provided samples, or if NO sample is given,
                                       b) The decoded output of a random latent space sample
        """
        if sample is not None:
            # Turn input to list, if not already
            if not isinstance(sample, list):
                sample = [sample]
            # If input is not in [batachSize,channels, h,w] format, and its just an image like (channels,h,w) add the batch dimension
            if len(sample[0].shape) < 4:
                for i in range(len(sample)):
                    sample[i] = sample[i].unsqueeze(0)
            # Preassign a tensor to hold the sum of the samples representations, the average of which will be used to generate a new fruit!
            genData = torch.zeros((sample[0].shape[0], self.latentDim)).to(self.device) 
            # Get a representation for all samples, average it and decode this average to get a new synthesized attempt!
            for i in range(len(sample)):
                genData += self.encode(sample[i])
            genData = self.decode(genData/len(sample))
            latentSample = 0
        else: #if latent sample is required
            latentSample = torch.rand((bSize, self.latentDim)).to(self.device)
            genData = self.decode(latentSample)
            
        return genData, latentSample
    # ------------------
    def report(self, **kwargs):
        """ DESCRIPTION: This customizable function serves a printout report for the networks progress, details etc.
                         Can be designed to suit any needs!   
        """
        print("Current stats of MNIST_NET:")
        print("Accuracy:      {}" .format(self.accuracy))
        print("Training Loss: {}" .format(self.trainLoss))
        print("Test Loss:     {}" .format(self.testLoss))

    # ====================================================================
 
