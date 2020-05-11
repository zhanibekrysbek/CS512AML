import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import copy
import sys
import os
from os.path import join
from random import shuffle
import utils
from plotter import plot_autoencoder
from utils import MSEReconstructionLoss
from indexes import PIDX as pidx
from indexes import  EIDX as eidx
import trainer
# Add this files directory location, so we can include the index definitions
dir_path = os.path.dirname(os.path.realpath(__file__))

# add this so we can import the trainer and tester modules
tools_path = os.path.join(dir_path, "../")
sys.path.insert(0, tools_path)
import indexes
# End Of imports -----------------------------------------------------------------

sign = lambda x: ('+', '')[x < 0]
# =====================================================================================================================
# Class Start.
# =====================================================================================================================

# CapWords naming convention for class names
# If there is a n acronym such as ANN, it is all uppercase
class AutoEncoderFrame(nn.Module):
    ''' Description:  This class provide the basic template for regressor networks.
        It provides the functions necessary for training, using, visualing and
        and logging.
        Inputs: encoder: (nn.module) The actual encoding network
                args: (variable args) General positional arguments that might be
                necessary for each encoder. They much be handled be each encoder
                Implementation
                kwargs: (dictionary) kew word arguments necessary for the encoder.
    '''

    def __init__(self, encoder, *args, **kwargs):
        super(AutoEncoderFrame, self).__init__()
        self.targetApp = 'Fruit_Generation' 
        # Argument handler/ sanitizer. Asigns values to self.kwargs/endKwargs
        self.handle_input_kwargs(*args,**kwargs)
        # ===============================================================================
        # SECTION A.0
        # ***********
        # NOTE: Change this appropriately for new Architectures!
        #
        # *******************************************************
        # Declare Name here. Used for saving and annotation.
        self.descr =  encoder.descr
        print(self.descr)
        # ---|

        # Declare the layers here
        if 'seed' in  kwargs:
            torch.manual(self.kwargs['seed'])
        # Use the provided encodera as the actual network arch that will engage in the
        # learning process.
        self.encoder = encoder
        # ---|

        # *******************************************************
        # Souldn't alter beneath this point for testing!!
        # ===============================================================================

        self.classMethod = encoder.classMethod # Can be either similarity, distance or label
        # The list below holds any plotted figure of the model
        self.plots = [None] * pidx.plotSize
        self.history = [[] for i in range(eidx.logSize)]
        self.predHistory = [[] for i in range(eidx.predLogSize)]
        # ---|
        # Default File Saving parameter setting.
        self.sep = '-'         # seperator for itemes in save file labels: i.e label1-secondLabel.txt if sep = '-'   
        self.rootSaveFolder    = os.path.join(dir_path, self.targetApp, self.descr)
        self.saveLogsFolder    = os.path.join(self.rootSaveFolder, 'Logs')
        self.saveResultsFolder = os.path.join(self.saveLogsFolder, 'Results')
        self.saveHisFolder     = os.path.join(self.saveLogsFolder, 'History')
        self.savePlotsFolder   = os.path.join(self.rootSaveFolder, 'Plots')
        self.saveModelsFolder  = os.path.join(self.rootSaveFolder, 'Models')
        self.saveSamplesFolder = os.path.join(self.saveLogsFolder, 'GenData')
        
        self.defSavePrefix = self.sep.join((self.descr,str(self.lr),str(self.momnt), str(self.wDecay)))
        self.info = self.sep.join((self.descr, self.defSavePrefix))
        self.defPlotSaveTitle = self.sep.join((self.descr, self.optim,"lr", str(self.lr),"momnt", str(self.momnt),
                                          'wDecay',str(self.wDecay), self.loss))
        # End of init
        # ---------------------------------------------------------------------------------

    def forward(self, x):
        # =================================================================================
        # Section A.1
        # ***********
        # Define the architecture layout of the model via an encoder. Encoders are defined
        # in a different py file, relevant to each problem category, i.e regressor_nets have
        # architecture relevant to regression problems.
        # *********************************************************************************
        x = self.encoder(x)
        return x
    
    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, x):
        return self.encoder.decode(x)

    def get_weights(self):
        return self.encoder.get_weights()

    def get_parameters(self):
        return self.encoder.get_parameters()
    
    #-------------------------------------------------------------------------------------------------------------------------------
    def generate(self, inExamples= None, fromLatentSpaceSample= False, saveFolder= None, saveTitle='GenSamples',**kwargs):
        
        if inExamples is not None:
            genData, _ = self.encoder.generate([i.to(self.encoder.device) for i in inExamples], **kwargs)
            printMat = torch.cat(inExamples, 0) 
            # Make the print image matrix to feed into mage_grid. WIll print the images in the following manner:
            # The first m rows will be the inExample images. The bottom row will be the generated result. Each row will have
            # batchSize images. The idea is that more images can fit the screen horizontally. Remember inExamples can be a list.
            print(printMat.shape)
            printMat = torch.cat([printMat, genData], 0)
        else:
            # In this caae the printMatrix is simply the decoded latent space samples. Untill we find a way to display differently sized
            # images together.
            genData, latentSample = self.encoder.generate(inExamples, **kwargs)
            printMat = genData
        if saveFolder is not None:
            saveSampleFile = os.path.join(saveFolder, 'Samples')
        else:
            saveSampleFile = self.saveSamplesFolder
        if not os.path.exists(saveSampleFile):
            os.makedirs(saveSampleFile)
        
        
        imgsPerRow = genData.shape[0]
        gridImg  = make_grid(printMat, nrow = imgsPerRow)
        saveSamplePath =os.path.join(saveSampleFile,saveTitle)
        save_image(gridImg,'{}.png'.format(saveSamplePath), nrow = imgsPerRow)
        '''
        for c in range(genData):
            sampleC = sample[:,:,c,:,:].view(-1,2,1,self.dataShape[0],self.dataShape[1])
            # sample = sample[:,:,0,:,:].view(-1,2,0,28,28)
            print("Reshaped sample shape: {}".format(sample.shape))
            img_grids = [make_grid(sampleC[:,v,:,:,:], nrow=stepSize) for v in range(self.numOfVae)]
            print("Image grid  shape: {},{}".format( len(img_grids),img_grids[0].shape))
            # print(img_grids[0][0])
            catted = torch.cat(img_grids, 1)
            print("Catteed  shape: {}".format( catted.shape))
            title = 'sampleZ' + 'channel-' + str(c)
            saveSamplePath = '/'.join((self.defSavePath,'Logs', self.descr,saveRootFolder, 'Samples'))
            if not os.path.exists(saveSamplePath):
                os.makedirs(saveSamplePath)
            saveSamplePath = '/'.join((saveSamplePath,title))
            save_image(catted,'{}.png'.format(saveSamplePath, title), nrow=stepSize)
        
        ''' 
               
        
        
        return genData

    #-------------------------------------------------------------------------------------------------------------------------------
    def print_layers(self):
        l = [module for module in self.encoder.modules() if type(module) != nn.Sequential]
        print(l)
        return l
    #-------------------------------------------------------------------------------------------------------------------------------

    def get_model_descr(self):
        """Creates a string description of a polynomial."""
        model = 'y = '
        return model
    
    def update_weight_holder(self, outputs, labels):
        pass
    #-------------------------------------------------------------------------------------------------------------------------------

    def save_history(self, filePath = None, savePredHist = False, saveTrainHist = True, saveResults = False, results = None,
                        resultsLabel = '', historyLabel = ''):

        sep = self.sep
        resultsLabel = resultsLabel if resultsLabel is not '' else sep.join((self.descr, 'results'))
        if filePath is not None:
            rootBasePath  = filePath
            saveResFolder = os.path.join(saveFile,'Logs', 'Results')
            saveHisFolder = os.path.join(saveFile,'Logs', 'History')
        else:
            rootBasePath  = self.rootSaveFolder
            saveResFolder = self.saveLogsFolder
            saveHisFolder = self.saveHisFolder
        folders = [saveResFolder, saveHisFolder]
        # Create the Target Directory if does not exist.
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
        if saveResults == True:
            saveResFile = os.path.join(saveResFolder, sep.join((self.defSavePrefix, resultsLabel, ".txt")))
        saveFile = os.path.join(saveHisFolder, sep.join((self.defSavePrefix, "log1.txt")))

        # Save training history or predHistory as required.
        if saveTrainHist == True:
            utils.save_log(saveFile, self.history)
        if savePredHist == True :
            utils.save_log(saveFile, self.predHistory)
        # Save Results if required
        if saveResults == True:
            if results is not None:
                try:
                    utils.save_tensor(results, filePath = saveResFile)
                except (AttributeError, TypeError):
                    raise AssertionError('Input Results variable should be Tensor.')
            else:
                print("No Results Tensor to save is given.")
        return saveFile
    # ---------------------------------------------------------------------------------------------------------
    def report(self):
        return self.encoder.report()
    #-------------------------------------------------------------------------------------------------------------------------------

    def fit(self, trainLoader, valLoader, optim, device, epochs =10, lossFunction= nn.MSELoss(),  
            recErrorFunc = MSEReconstructionLoss(), validate = True, # basic arguments for fitting and validation enabling
            adaptDurationTrain=False, printInterval=40, earlyStopIdx=0, earlyTestStopIdx=0, classMethod ='distance',   # arguments for train/test stopage control and printouts
            saveHistory = False, savePlot = False, modelLabel ='',  # arguments for saving
            w=10, epsilon = 0.03, verbose = True,  # arguments for handing dynamic convergence checking
            **kwargs):                                              # various arguments that downstream training and testing function might need.
        """ DESCRIPTION: THis function handle the iterative process of updating a model's parameters to match the data; "fitting". It will handle
                         the train and, if need be, validation on given data. The function is a wrapper for the the train and testing function
                         where the actual heavy lifting is done, in the trainer.py file. This function will pass any appropriate arguments to those functions,
                         handle early stopping, dynamic convergence check (if performance stalls, it will exit see dynamic_conv_check for more info) and can also
                         save the results, history and plots of the fitting process. This structure can allow for different training/ testing functions to be used
                         without moving around too much code. See the train and test function in trainer.py for more info on how they handle things.
            ARGUMENTS:
            
            RETUNRS: None
        """
        epochs = epochs
        # Create deep copies of the arguments, might be required to the change.
        avgTrL = 0
        tlTrL = 0
        
        # Turn input arguments in input dictionaries for the testing function
        trainerArgs = dict(trainerArgs=dict(classMethod=classMethod,epoch=1, printInterval= printInterval,
                                            stopIdx=earlyStopIdx))
        testerArgs  = dict(testerArgs=dict(classMethod=classMethod, trainMode = True, epoch =1,
                                           earlyStopIdx=earlyTestStopIdx, printInterval=printInterval))
        trainerInputArgs= {**dict(lossFn=lossFunction, optim=optim, recErrorFunc=recErrorFunc), **trainerArgs}
        testerInputArgs = {**dict(lossFn=lossFunction),**testerArgs}
        # ---|
        
        scheduler = lr_scheduler.ReduceLROnPlateau(optim, verbose=verbose, patience =4, eps=1e-9)
        # scheduler = lr_scheduler.CyclicLR(optim, 1e-5, 5*1e-4, step_size_up=401)

        # Training Loop
        # ======================================================
        # For each epoch train and then test on validation set.
        for e in range(epochs):
            # Pre-cycle handling of arguments
            trainerInputArgs['trainerArgs']['epoch'], testerInputArgs['testerArgs']['epoch'] = e, e
            # Learning scheduler has to be called like so
            if self.encoder.lr_policy  == 'plateau':
                scheduler.step(self.encoder.metric)
            else:
                scheduler.step()
            # ---|

            # Train and validate
            # ****
            avgTrL, tlTrL = trainer.train_autoencoder(self, trainLoader, device, **trainerInputArgs)
            if validate:
                output, loss = trainer.test_autoencoder(self, valLoader, device, **testerInputArgs)
            # ***|

            # Post Cycle Handling
            # If adaptive duration for training is enabled, then break training process if progress
            # in loss is less than epsilon. Should be expanded in average over past few epochs.
            if adaptDurationTrain == True:
                if trainer.dynamic_conv_check(self.history, args=dict(window= w, percent_change=epsilon,
                                                                      counter =e, lossIdx=e.testLoss)):
                    break
                else:
                    print("CONTINUE")
        # ======================================================
        
        # If saving history and plots is required.
        modelLabel = modelLabel if modelLabel is not None else self.defSavePrefix + "-for-"+str(epochs)+'-epchs'
        if saveHistory == True:
            self.save_history(historyLabel = modelLabel)
            print("Saving model {}-->id: {}".format(self.defPlotSaveTitle, hex(id(self))))

        # If no args for tarFolder are given plots go to the preTrain folder.
        # As: architect-0-optimName-lr-x-momnt-y-wD-z-LossName.png
        if savePlot == True:
            self.plot(mode='All')
            self.save_plots()

    #----------------------------------------------------------------------------------------------------

    # Testing and error reports are done here
    def predict():
        pass
   
    #-------------------------------------------------------------------------------------------------------------------------------

    def plot(self, mode = 'Learning Curves', source = 'Logs', logPath = None, title = ''):
        ''' Description: This function is a wrapper for the appropriate plot function Found in the Tools package. It handles any architecture spec
                         cific details, that the general plot function does not, such
                         as: how many logs to read  and plot in the same graph.
            Arguments:   logPath  (string): Location of the log files to read. If None, function will read the logs in the default location.
                         mode (String): The plotting mode. Can be Learning Curves for the moment.  
                         title (String): A title for the plot. If left '', the default title is formed as defined in self.defPlotTitle
        '''
        # Args is currently empty. Might have a use for some plotting arguments
        # In the future. Currently none are implemented.
        args = []
        if source == 'Logs':
            if logPath is not None:
                logPath = logPath
            else:
                logPath = os.path.join(self.saveHisFolder, self.sep.join((self.defSavePrefix, "log1.txt")))
            # Form plot title and facilate plotting
            title = self.descr + ' '+'Learning Curves'
            self.plots[pidx.lrCurve] = plot_autoencoder(filesPath = logPath, title = title, mode = 'Learning Curves')
            title = self.descr + ' '+'Reconstruction Error'
            self.plots[pidx.recError] = plot_autoencoder(filesPath = logPath, title = title, mode = 'Recon Error')
        elif source == 'History':
            if mode == 'Learning Curves' or mode == 'All':
                title = self.descr + ' Learning Curves'
                self.plots[pidx.lrCurve] = plot_autoencoder(inReps = self.history,title=title, mode='Learning Curves')
            if mode == 'Recon Error' or mode == 'All':
                title = self.descr+' Reconstruction Error'
                self.plots[pidx.recError] = plot_autoencoder(inReps = [self.history], title=title, mode='Recon Error')

    #-------------------------------------------------------------------------------------------------------------------------------

    # Save plots
    def save_plots(self, savePath = None, plotLabel = '', defTitleExt = ''):
        '''DESCRIPTION: This function saves all plots of model. If no target path is given, the default is selected.
                        The default is the PreTrain folder of target architecture and application.
           ARGUMENTS: savePath (Path): The target folder that the plots will be saved. If None the default Folder will be used.
                      plotLabel (String): The complete base label for the plots. A number will be added to avoide overwrites. 
                      fileExt (String): To be used as an addition to the default save names for the plot images. NOTE: if you want a 
                                        compltely different filename than the default, use plotLabel instead.
        '''
        if savePath is None:
            #savePath = os.path.join(self.defSavePath, 'Plots', self.descr, saveRootFolder, tarFolder)
            savePath = self.savePlotsFolder
            # Create the Target Directory if does not exist.
        if not os.path.exists(savePath):
            os.makedirs(savePath)
                
        os.chdir(savePath)
        for i, f in enumerate(self.plots):
            if f is not None:
                if plotLabel is not '':
                    fileName = self.sep.join((plotLabel, str(i)+'.png'))
                else:
                    fileName = self.sep.join( (self.defPlotSaveTitle, str(i), defTitleExt+'.png') )
                print("*****\nSaving figure: {} at {}****\n".format(self.descr, savePath +"/" + fileName ))
                curDir = os.getcwd()
                f.savefig(fileName)
        os.chdir(curDir)
                
    #-------------------------------------------------------------------------------------------------------------------------------

    def save(self, savePath = None, modelLabel =''):

        if savePath is None:

            savePath =elf.saveModelsFolder
            # Create the Target Directory if does not exist.
            if not os.path.exists(savePath):
                os.makedirs(savePath)
        modelLabel = modelLabel if modelLabel is not '' else self.defPlotSaveTitle
        savePath = os.path.join(savePath, modelLabel)
        print("****\nSaving model: {}-->id: {} at {}\n****".format(self.descr, hex(id(self)), savePath))

        #utils.save_model_dict(self.encoder, savePath+fileExt)
        torch.save(self.encoder.state_dict(), savePath)
    #-------------------------------------------------------------------------------------------------------------------------------

    def load(self, loadPath = None):

        if loadPath is None:
            loadPath = os.path.join(self.saveModelsFolder, self.defPlotSaveTitle)
            
            # Create the Target Directory if does not exist.
        if not os.path.exists(loadPath):
            print("Given path or model does not exists. Will try to load from defaualt location.")
        else:
            print("Loading saved model: {} to model {}@{}".format(loadPath, self.descr, hex(id(self))))

        self.load_state_dict(torch.load(loadPath)) 

    #-------------------------------------------------------------------------------------------------------------------------------

    def print_out(self, mode='history'):

        if mode == 'History':
            print("Current stats of ANNSLF:")
            print("Accuracy:           {}" .format(self.history[cidx.trainAcc][-1]))
            print("Recall:          {}" .format(self.history[cidx.trainRec][-1]))
            print("Training Loss: {}" .format(self.history[cidx.trainLoss][-1]))
            print("Test Acc:      {}" .format(self.history[cidx.testAcc][-1]))
            print("Test Recall:     {}" .format(self.history[cidx.testRec][-1]))
            print("Test Loss:     {}" .format(self.history[cidx.testLoss][-1]))
        if mode == 'params':
            print(list(self.parameters()))

    #-------------------------------------------------------------------------------------------------------------------------------
    def handle_input_kwargs(self,*args, **kwargs):
        # parameters here hold info for optim used to train the model. Used for annotation.
        # Save the arguments used to create this Frame. Can be later used for annotation
        # and copying
        self.args = args
        print(kwargs)
        if bool(kwargs):
            self.kwargs   = kwargs['templateKwargs'] if 'templateKwargs' in kwargs.keys() else None
            self.encoderKwargs= kwargs['encoderKwargs'] if 'encoderKwargs' in kwargs.keys() else None

            self.loss  = 'defLoss'  if 'loss'  not in self.kwargs.keys() else self.kwargs['loss']
            self.optim = 'defOptim' if 'optim' not in self.kwargs.keys() else self.kwargs['optim']
            self.lr    = 'defLr'    if 'lr'    not in self.kwargs.keys() else self.kwargs['lr']
            self.momnt = 'defMomnt' if 'momnt' not in self.kwargs.keys() else self.kwargs['momnt']
            self.wDecay= 'defwDecay'if 'wDecay'not in self.kwargs.keys() else self.kwargs['wDecay']
            self.w     = 'defW'     if 'w'     not in self.kwargs.keys() else self.kwargs['w']
            self.targetApp = 'defApp' if 'targetApp' not in self.kwargs.keys() else self.kwargs['targetApp']
    #-------------------------------------------------------------------------------------------------------------------------------
    def create_copy(self, device, returnDataShape = 0):
        ''' Description: This function will create a ne deep copy of this model.
            Arguments:  device:          Where the model is instantiated, GPu or CPU
                        returnDataShape: Flag of whether a return string is required for raw data
                                         reshaping
            returns:    model: A model object, with all the parameters of the initial one, deep-copied.
        '''
        # Create a clone of the encoding network this regressor uses.
        if 'ANN' in self.descr:
            state_clone = copy.deepcopy(self.encoder.state_dict())
            encoder = type(self.encoder)(*self.args, **self.encoderKwargs).to(device)
            encoder.load_state=dict(state_clone)
        else:
            embeddingNet = self.encoder.embedding_net
            state_clone = copy.deepcopy(embeddingNet.state_dict())
            encoder = type(self.encoder)(embeddingNet,*self.args, **self.encoderKwargs).to(device)
            encoder.load_state=dict(state_clone)
        # ---|
        # Create a copy of the regressor template
        kwargs = dict(templateKwargs=self.kwargs,encoderKwargs=self.encoderKwargs)
        model = type(self)(encoder, self.args, **kwargs).to(device)

        reShapeDataTo = self.descr
        if returnDataShape == 1:
            return model, reShapeDataTo
        else:
            return model
