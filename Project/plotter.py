import sys
import os
# from pathlib import Path
# from os.path import isdir, join, isfile
# from os import listdir
# import fnmatch
# import re
# import torch.nn as nn
# from dateutil.parser import parse
import matplotlib.pyplot as plt
from random import randint
from matplotlib import markers
import numpy as np
from itertools import cycle
from indexes import CIDX as cidx
from indexes import EIDX as eidx
import math


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



def plot_autoencoder(filesPath=None, title = '', xAxisNumbers = None, labels=[], inReps = [], plot = 'All', mode = 'Learning Curves'):
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

    if not isinstance(filesPath, list) and filesPath is not None:
        files = [filesPath]
    else:
        files = filesPath
    reps = []

    if filesPath:
        for i,f in enumerate(files):
            reps.append([[] for i in range(eidx.logSize)])
            # print(i)
            # print("Size of reps list: {} {}".format(len(reps),len(reps[i])))
            with open(f, 'r') as p:
                # print("i is {}".format(i))
                for j,l in enumerate(p):
                    # Ignore last character from line parser as it is just the '/n' char.
                    report = l[:-2].split(' ')
                    # print(report)
                    reps[i][eidx.trainLoss].append(report[eidx.trainLoss])
                    reps[i][eidx.testLoss].append(report[eidx.testLoss])
                    reps[i][eidx.trainRecErr].append(report[eidx.trainRecErr])
                    reps[i][eidx.testRecErr].append(report[eidx.testRecErr])
        epochs = len(reps[0][0])


    if inReps:
        for i,r in enumerate(inReps):
            reps.append(r)
        epochs = len(reps[0][0])
    # print("Plots epochs: {}" .format(epochs))
    # iac.breakpoint(True)
    if mode == 'Learning Curves':
        xLabel = 'Epoch'
        yLabel = 'Loss'
        pIdx1, pIdx2  =  eidx.trainLoss, eidx.testLoss
    elif mode == 'Recon Error':
        xLabel = 'Epoch'
        yLabel = 'Reconstruction Error'
        pIdx1, pIdx2  =  eidx.trainRecErr, eidx.testRecErr
    
    if xAxisNumbers is None:
        epchs = np.arange(1, epochs+1)
    else:
        epchs = xAxisNumbers
    # ---|

    fig = plt.figure(figsize=(19.2,10.8))
    # fig = plt.figure(figsize=(13.68,9.80))
    plt.title(title)
    plt.ylabel(yLabel)
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
        #print(a, epchs)
        # WHen plotting multiple stuff in one command, keyword arguments go last and apply for all
        # plots.
        # If labels are  not given make a generic label from the infile name or from the VAE position
        if not labels:
            if files:
                print(files)
                ext = os.path.split(files[i])[1].split('-')
                ext = ' '.join(('lr', ext[0],'m',ext[1],'wD',ext[2]))
            else:
                ext = ''
        else:
            ext = labels[i]
            print(ext)
        # Select color for the plot
        cSel = [randint(0, len(cmaps)-1), randint(0, len(cmaps)-1)]
        c1 = plt.get_cmap(cmaps[cSel[0]])
        # Solid is Train, dashed is test
        marker = markerList[randint(0, len(markerList))]
        if plot == 'All' or plot == 'Train':
            plt.plot(epchs, a[pIdx1], color = c1(i / float(len(reps))), linestyle ='-', marker=marker, label = 'Train-'+ext)
        # plt.plot(epchs, a[eidx.testLoss],  (str(next(cycol))+markerList[rndIdx]+'--'), label = ext)
        if plot == 'All' or plot == 'Test':
            linestyle = "-" if "no" in ext else "--"
            linestyle = ":" if "provided" in ext else linestyle
            plt.plot(epchs, a[pIdx2], color=  str(next(cycol)), linestyle = linestyle, marker=marker, label = 'Test-'+ext)
        plt.legend( loc='upper right')
        ext_list.append(ext)
        test_loss_list.append(a[eidx.testLoss][-1])

    best_index = np.argmin(np.array(test_loss_list))
    print("Best test loss is:", str(test_loss_list[best_index]))
    print("Best parameters are:", ext_list[best_index])
        # plt.close()
        # plt.draw()
        # plt.pause(15)

    return fig

# -----------------------------------------------------------------------------------------------------------------------------
def display_tensor_image( tensorImage):
    """ DESCRIPTION: Visualize tensor as an image.
        ARGUMENTS: tensorImage (tesnor): Tensor to be siplayed. Should be in Heighxt x Width x Channles format.
        RETURNS: NONE
    """
    plt.imshow(  tensorImage  )
    plt.show()