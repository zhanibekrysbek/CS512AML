import sys
import os
from pathlib import Path
from os.path import isdir, join, isfile
from os import listdir
import fnmatch
import re
import torch.nn as nn
from dateutil.parser import parse
import matplotlib.pyplot as plt
from random import randint
from matplotlib import markers
import numpy as np
from itertools import cycle
from indexes import CIDX as cidx
import math

# -------------------------------------------------

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

# -------------------------------------------------
class shapedUnFlatten(nn.Module):
    def __init__(self, channels = 1, height = 1, width = 1):
        super(shapedUnFlatten, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, inp):
        # print(inp.shape)
        # red = inp.view(inp.size(0), -1, 58 , 58)
        # print(red.shape)
        # return red
        return inp.view(inp.size(0), self.channels, self.height, self.width)
# -------------------------------------------------
class UnFlatten(nn.Module):

    def forward(self, inp):
        return inp.view(inp.size(0), -1, 1, 1)

# -------------------------------------------------
# Layer for debugging nn.Sequential packages
class Print(nn.Module):
    def __init__(self, enable = False, div=False):
        super(Print, self).__init__()
        self.div    = div
        self.enable = enable

    def forward(self, x):
        if self.enable:
            if self.div:
                print("--------------")
            print(x.shape)
        return x
# -------------------------------------------------
def show(img):
    ''' DESCRIPTION: THis function will display a tensor image of format(C,H,W) which is the numpy format.
    '''
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
# -------------------------------------------------
    
def plot(history, eps = None, train_test=1, figType = 'acc', saveFile = None, title='Test Accuracy vs Epoch'):
    if figType == 'acc':
        fig = plt.figure(figsize=(10, 6), dpi=200)
        plt.title('Plain LSTM accuracy')
        plt.xlabel('Epochs')
        if train_test == 0:
            plt.ylabel('Train Accuracy')
        else:      
            plt.ylabel('Test Accuracy')

        xAxis = np.arange(len(history[1]))+1
        plt.plot(xAxis, history[train_test], marker = 'o')
        plt.grid()
        if saveFile is not None:
            plt.savefig(saveFile)
        return fig

    if figType == 'lrCurve':
        fig = plt.figure(figsize=(10, 6), dpi=200)
        plt.title(title)
        plt.xlabel('Epochs')
        if train_test == 0:
            plt.ylabel('Train Accuracy')
        else:      
            plt.ylabel('Test Accuracy')
        for ind,hist in enumerate(history):
            xAxis = np.arange(len(hist[train_test]))+1
            plt.plot(xAxis, hist[train_test], marker = 'o', alpha=0.9, label=f"eps={eps[ind]}")
        plt.legend()
        plt.grid()
        if saveFile is not None:
            plt.savefig(saveFile)
        return fig

# -----------------------------------------------------------------------------------------

def get_files_from_path(targetPath, expression, excludePattern = 'dumz'):

    # Find all folders that are not named Solution.
    d = [f for f in listdir(targetPath) if (isdir(join(targetPath, f)) and "Solution" not in f)]
    # Find all file in target directory that match expression
    f = [f for f in listdir(targetPath) if (isfile(join(targetPath, f)) and fnmatch.fnmatch(f,expression) and excludePattern not in f)]
    # initialize a list with as many empty entries as the found folders.
    l = [[] for i in range(len(d))]
    # Create a dictionary to store the folders and whatever files they have
    contents = dict(folders=dict(zip(d,l)))
    contents['files'] = f

    # Pupulate the dictionary with files that match the expression, for each folder.
    # This will consider all subdirectories of target directory and populate them with
    # files that match the expression.
    for folder, files in contents.items():
        stuff = sorted(Path(join(targetPath, folder)).glob(expression))
        for s in stuff:
            files.append(os.path.split(s)[1] )
        # print(folder, files)
    for files in contents['files']:
        stuff = sorted(Path(join(targetPath, files)).glob(expression))
    # print(contents)
    return contents


# -----------------------------------------------------------------------------------------
def save_log(filePath, history):
    '''
        Description: Saves the history log in the target txt file.
                     If some history elements do not exist, mark them with -1.
        Arguments:   filePath (string): Target location for log
                     history (list of lists): History list in the following format:
                     Each  inner list is one of trainMAE, testLOss etc, as indexed
                     in the ridx file. They contain the relevant metric from all epochs
                     of training / testing, if they exist.
    '''

    with open(filePath, 'w') as f:
        for i in range(len(history[0])):
            for j in range(len(history)):
                if history[j] and j < len(history):
                    f.write("{:.4f} ".format(history[j][i]))
                else:
                    f.write("-1")
            f.write("\n")

            
def plot_classifier(filesPath='', title = '', xAxisNumbers = None, labels=[], inReps = [], plot = 'All', mode = 'Learning Curves'):
    ''' Description: This function will plot Learning or Prediciton curves, as supplied from either txt log files, or a list of
                     histories, or both. It returns a figure, containg all the curves; one curve for each history provided.
        Arguments:  filesPath(filePath): A file path to the folder containing the required log txt files.
                    title(String):       Title to the figure
                    xAxisNumbers(List):  A list of lalbel strings to be used as x axis annotations.
                    labels(List):        A list of strings to be used as curve labels.
                    inReps(List):        A list of model histories in the format train-loss MAE MAPE test-MAE MAPE loss.
                    plot (selector)      A string command  not yet offering functionality
                    mode(Selector):      A string command telling the function to plot Learning curves or simple prediction loss.
        Returns:    fig:     A figure object containg the plots.
    '''
    # Argument Handler
    # ----------------------
    # This section checks and sanitized input arguments.
    if not filesPath and  not inReps:
        print('No input log path or history lists are given to plot_regressor!!')
        print('Abort plotting.')
        return -1

    if not isinstance(filesPath, list):
        files = [filesPath]
    else:
        files = filesPath
    reps = []

    if filesPath:
        for i,f in enumerate(files):
            reps.append([[] for i in range(cidx.logSize)])
            # print(i)
            # print("Size of reps list: {} {}".format(len(reps),len(reps[i])))
            with open(f, 'r') as p:
                # print("i is {}".format(i))
                for j,l in enumerate(p):
                    # Ignore last character from line parser as it is just the '/n' char.
                    report = l[:-2].split(' ')
                    # print(report)
                    reps[i][cidx.trainAcc].append(report[cidx.trainAcc])
                    reps[i][cidx.trainAcc5].append(report[cidx.trainAcc5])
                    reps[i][cidx.trainLoss].append(report[cidx.trainLoss])
                    reps[i][cidx.testAcc].append(report[cidx.testAcc])
                    reps[i][cidx.testAcc5].append(report[cidx.testAcc5])
                    reps[i][cidx.testLoss].append(report[cidx.testLoss])

    if inReps:
        for i,r in enumerate(inReps):
            # reps.append([[] for i in range(ridx.logSize)])
            reps.append(r)
    # print("Plots epochs: {}" .format(epochs))

    epochs = len(reps[0][0])
    if mode == 'Learning Curves':
        xLabel = 'Epoch'
    elif mode == 'Prediction History':
        xLabel = 'Task'

    if xAxisNumbers is None:
        epchs = np.arange(1, epochs+1)
    else:
        epchs = xAxisNumbers
    # ---|

    fig = plt.figure(figsize=(19.2,10.8))
    # fig = plt.figure(figsize=(13.68,9.80))
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel(xLabel)
    # Set a color mat to use for random color generation. Each name is a different
    # gradient group of colors
    cmaps= ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                     'Dark2', 'Set1', 'Set2', 'Set3',
                     'tab10', 'tab20', 'tab20b', 'tab20c']
    # Create an iterator for colors, for automated plots.
    cycol = cycle('bgrcmk')
    ext_list = []
    test_loss_list = []
    markerList = list(markers.MarkerStyle.markers.keys())[:-4]
    for i, rep in enumerate(reps):
        # print(cmap(i))
        a = np.asarray(rep, dtype = np.float32)
        # WHen plotting multiple stuff in one command, keyword arguments go last and apply for all
        # plots.
        # If labels are given
        if not labels:
            ext = os.path.split(files[i])[1].split('-')
            ext = ' '.join(('lr', ext[0],'m',ext[1],'wD',ext[2]))
        else:
            ext = labels[i]
            print(ext)
        # Select color for the plot
        cSel = [randint(0, len(cmaps)-1), randint(0, len(cmaps)-1)]
        c1 = plt.get_cmap(cmaps[cSel[0]])
        # Solid is Train, dashed is test
        marker = markerList[randint(0, len(markerList))]
        if plot == 'All' or plot == 'Train':
            plt.plot(epchs, a[cidx.trainLoss], color = c1(i / float(len(reps))), linestyle =
                 '-', marker=marker, label = 'Train-'+ext)
        # plt.plot(epchs, a[ridx.testLoss],  (str(next(cycol))+markerList[rndIdx]+'--'), label = ext)
        if plot == 'All' or plot == 'Test':
            linestyle = "-" if "no" in ext else "--"
            linestyle = ":" if "provided" in ext else linestyle
            plt.plot(epchs, a[cidx.testLoss], color=  str(next(cycol)), linestyle = linestyle, marker=marker, label = 'Test-'+ext)
        plt.legend( loc='upper right')
        ext_list.append(ext)
        test_loss_list.append(a[cidx.testLoss][-1])

    best_index = np.argmin(np.array(test_loss_list))
    print("Best test loss is:", str(test_loss_list[best_index]))
    print("Best parameters are:", ext_list[best_index])
        # plt.close()
        # plt.draw()
        # plt.pause(15)

    return fig

#----------------------------------------------------------------------------------------------

def comp_pool_dimensions(layerType,height, width, kSize, depth = 0, padding=0, dilation=1, stride=1,retType = 'list'):
    dims = int(''.join(filter(str.isdigit, str(layerType))))

    if type(padding) is not (list and tuple):
        padding = [padding, padding]
    if type(dilation) is not (list and tuple):
        dilation = [dilation, dilation]
    if type(kSize) is not (list and tuple):
        kSize = [kSize, kSize]
    if type(stride) is not (list and tuple):
        stride = [stride, stride]


   
    heightOut = math.floor(int(((height+ 2*padding[0] - dilation[0] * (kSize[0]-1)-1) / stride[0]) +1))
    widthOut  = math.floor(int(((width + 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1))
    if dims == 3:
        depthOut  = int(((depth + 2*padding[0] - dilation[2] * (kSize[0]-1)-1) / stride[0]) +1)
        heightOut = int(((height+ 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1)
        widthOut  = int(((width + 2*padding[2] - dilation[2] * (kSize[2]-1)-1) / stride[2]) +1)

    if retType == 'list':
        return [heightOut, widthOut] if dims != 3 else [depth,height, width]
    else:
        return  dict(height=heightOut, width = widthOut) if dims != 3 else dict(depth=depthOut, height=heightOut, width = widthOut)
    
# --------------------------------------------------------------------------------------------------------

def comp_conv_dimensions(layerType, height, width, kSize, depth = 0, padding=0, dilation=1, stride=1, outputPadding = 0,retType = 'list'):
    ''' DESCRIPTION: This function computes the out dimensions of any convolutional or transpose convolutional
                     pytorch layer. It returns a list or dict of the computed output dimensions of size 1 for
                     1D conv, size 2 for 2D and 3 for 3D.
        ARGUMENTS: layerType-> (type) type of this layer.
                   Rest of args: (int or list,tuple): Input to this layer in this order: height,width, kernel
                   size of layer. If the given inputs are not list, tuple the scalars are repeated to form the
                   required holder.
                   Rest of keword args: Similarly to args. These are usually set to the default values, hence
                   the keyword format.
    '''
    dims = int(''.join(filter(str.isdigit, str(layerType))))

    if type(padding) is not (list and tuple):
        padding = [padding, padding]
    if type(dilation) is not (list and tuple):
        dilation = [dilation, dilation]
    if type(kSize) is not (list and tuple):
        kSize = [kSize, kSize]
    if type(stride) is not (list and tuple):
        stride = [stride, stride]
    if type(outputPadding) is not (list and tuple):
        outputPadding = [outputPadding, outputPadding]


    if 'Transpose' in str(layerType):
        heightOut = math.floor(int( (height-1) * stride[0]  - 2*padding[0] + dilation[0] * (kSize[0]-1) + outputPadding[0] +1))
        widthOut  = math.floor(int( (width -1) * stride[1]  - 2*padding[1] + dilation[1] * (kSize[1]-1) + outputPadding[1] +1))
        if dims == 3:
            depthOut  = math.floor(int( (depth -1) * stride[0] - 2*padding[0] + dilation[0] * (kSize[0]-1) + outputPadding[0] +1))
            heightOut = math.floor(int( (height-1) * stride[1] - 2*padding[1] + dilation[1] * (kSize[1]-1) + outputPadding[1] +1))
            widthOut  = math.floor(int( (width -1) * stride[2] - 2*padding[2] + dilation[2] * (kSize[2]-1) + outputPadding[2] +1))
    else:
        heightOut = math.floor(int(((height+ 2*padding[0] - dilation[0] * (kSize[0]-1)-1) / stride[0]) +1))
        widthOut  = math.floor(int(((width + 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1))
        if dims == 3:
            depthOut  = math.floor(int(((depth + 2*padding[0] - dilation[2] * (kSize[0]-1)-1) / stride[0]) +1))
            heightOut = math.floor(int(((height+ 2*padding[1] - dilation[1] * (kSize[1]-1)-1) / stride[1]) +1))
            widthOut  = math.floor(int(((width + 2*padding[2] - dilation[2] * (kSize[2]-1)-1) / stride[2]) +1))

    if retType == 'list':
        return [heightOut, widthOut] if dims != 3 else [depth,height, width]
    else:
        return  dict(height=heightOut, width = widthOut) if dims != 3 else dict(depth=depthOut, height=heightOut, width = widthOut)

# ===================================================================================================================
# SCORES
# ===================================================================================================================

def compute_inception_score(lDist, mDist, verbose = False):
    score = 0
    # Add epsilon to marginal distribution(collection, really) to avoid inf
    mDist += 1
    # Normalize marginal, so it becomes a proper distribtuion.
    marginal = mDist / mDist.sum(dim=0)
    # Compute KL Divergence P(x) then Q(x)
    KLDivergence = torch.sum(lDist * (torch.log(lDist) - torch.log(marginal)), dim = 1)
    # score2 = entropy(lDist, mDist.unsqueeze(0).expand(lDist.size(0),-1,-1))
    score = torch.exp(KLDivergence.mean(dim=0))
    if verbose:
        print("Marginal p(y):", marginal)
        print("Marginal p(y):", mDist)
        print("label Distribution p(y|x):", lDist[:,:,1])
    return score

# ===================================================================================================================
# VISUALIZATION + SAVE N LOAD
# ===================================================================================================================

def save_tensor(tensor, delimeter = ' ', filePath = None):
    ''' Description: This function saves a tensor to a txt file. It first copies
                     it to host memory, turn it into a numpy array and dump it
                     into a txt file. This is faster that a for loop by an order
                     of magnitude. Original tensor stays in GPU.
        Arguments:  tensor (p Tensor): The tensor containing the results
                    to be written.
                    delimeter (String): A string thatwill separate data in txt
                    filePath (String): Target path to save file
    '''
    a = tensor.cpu()
    a = a.numpy()
    np.savetxt(filePath, a,  fmt="%.4f", delimiter=delimeter)
 

# --------------------------------------------------------------------------------------------------------------------

class BCE_KLD_CompoundLoss(nn.Module):
    """ Description: see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                     TODO: angle loss is an input currently. Has to become a normal loss. Talk to Aswin about
                     it.
    """
    def __init__(self):
        super(BCE_KLD_CompoundLoss, self).__init__()

    def forward(self, x, target, recon_x, mu, logvar, z, angleLoss=0, weights= None):
        if weights is not None:
            weights.detach()
        BCE = F.binary_cross_entropy(recon_x, x, reduction = 'none')
        BCE = torch.sum(BCE, dim=tuple(range(2, len(BCE.size()))))
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE+KLD+angleLoss

    def get_BCE(recon_x,x, size_average = False):
        BCE = F.binary_cross_entropy(recon_x, x)
        return BCE

    def get_KLD(mu, logvar):
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
# ==================================================================================================
# REGRESSION LOSSES
# ==================================================================================================

class MSEReconstructionLoss(nn.Module):
    ''' DESCRIPTION: THis function computes the mean square error between inut and target making sure
                     the target is sxpanded to the required dimensions
        ARGUMENTS: x: (Tensor) Input Tensor of shape either [Batch x N DImensionx Channel x Height x Width]
                   target: (Tensor) Target Tensor of shape [Batch x  Channel x Height x Width]
    '''
    def __init__(self):
        super(MSEReconstructionLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self,x,target):
        # print(x.shape, target.shape)
        if len(x.shape) > len(target.shape):
            target = target.unsqueeze(1).expand_as(x)
        else:
            target = target.expand_as(x)
        loss = self.loss(x,target)
        return loss
# -------------------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------------------
def main():
    multiThread = False 
    # get this files path
    dir_path= os.path.dirname(os.path.realpath(__file__))
    filePath = os.path.join(dir_path, "Data", "fruits-360_dataset", "fruits-360", "Training")
    f = get_files_from_path(filePath, "*.xlsx", excludePattern='filtered')
    print(f)
    # ---|
   

# -------------------------------------------------------------------

if __name__ == '__main__':
    main()