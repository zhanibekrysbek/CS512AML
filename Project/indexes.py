
class CIDX():
    """ DESCRIPTION: This class defines the list idnexes the various losses and acc are stored in the
        history variable of the classifier framework. For example all trainLosses, for all epochs are
        in model.history[0].
    """
    #This detrsmines the number of columns in the history attribute of each 
    # reggression model
    logSize     = 6
    predLogSize = 6  # fill with zeros first 3 cols for plotting compatibility

    # Train phase metrics
    trainLoss = 0
    trainAcc  = 1
    trainAcc5 = 2
    # Test phase metrics
    testLoss  = 3
    testAcc   = 4
    testAcc5  = 5

    # Prediction phase metrics
    predLoss = 6
    predAcc  = 7
    predAcc5 = 8
    
class PIDX():
    ''' DESCRIPTION: THis class defines indexes to be used for accessing the stored plots of a model
    ''' 
    plotSize    = 2
    encPlotSize = 4

    # MARE for regressors, accuracy for classifiers
    lrCurve     = 0
    predCurve   = 1

    # Indexers for variational autoencoders
    recError = 1
    itemsWeightHist = 2
    tasksWeightHist  = 3
    
class EIDX():  
    # This detersmines the number of columns in the history attribute of each 
    # reggression model
    logSize     = 4
    predLogSize = 4  # fill with zeros first 3 cols for plotting compatibility

    # Train phase metrics
    trainRecErr = 0
    trainLoss   = 1

    # Test phase metrics
    testRecErr = 2
    testLoss   = 3

    # Prediction phase metrics
    predRecErr  = 2
    predLoss = 3

    # Index of various lists
    ITEMS  = 0
    TASKS = 1