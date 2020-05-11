import torch.nn.functional as F
from indexes import CIDX as cidx
from indexes import EIDX as eidx
import torch
import ipdb
import utils



def train_classifier(model, indata, device, lossFn, optim,  **kwargs):
    
    verbose = True if 'verbose' not in kwargs.keys() else kwargs['verbose']
    epoch = kwargs['trainerArgs']['epoch'] if 'trainerArgs' in kwargs.keys() else -1
    stopIdx = kwargs['trainerArgs']['stopIdx'] if 'trainerArgs' in kwargs.keys() else 0 #used for early stopping at the target batch number
    printInterval = kwargs['trainerArgs']['printInterval'] if 'trainerArgs' in kwargs.keys() else 40
    factor = 0
    totalSize, totalLoss = 0, 0
    print("Train Device is {}".format(device))
    for idx, items in enumerate(indata):
        
        # A. Forward computation input retrieval and handling
        if type(items) == list:
            data   = items[0]
            target = items[1]
        else:
            target = items['target'].float() if any(items['target']) else None
            data = items['data']
            
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)
        
        if target is not None:
            target = target.to(device)
                
        # B. Forward pass calculate output of model
        output      = model.encoder.forward(*data)
                    
        # C. Loss computation part.
        # Convention for all loss and reconstruction inputs is Data, Target, miscInputs. Model forward MUST be
        # Designed to match its output to the loss functions' input pattern.
        if type(output) not in (tuple, list):
            output = (output,)
        # 1 position: data
        lossInputs = (output[0],)
        if target is not None:
            target = (target,)
        # 2 position target
            lossInputs += target
        # 3: positions-> rest of required misc Inputs to loss func.
        lossInputs += tuple(output[1:])
        
        # compute loss
        #ipdb.set_trace() # BREAKPOINT
        lossOutputs = lossFn(*lossInputs)
        loss  = lossOutputs[0] if type(lossOutputs) in (tuple, list) else lossOutputs
        #loss        = lossFn(output, label)
        totalLoss  += loss
        # loss        = F.CrossEntropyLoss(output, target)

        # D. Backpropagation part
        # 1. Zero out Grads
        optim.zero_grad()
        # 2. Perform the backpropagation based on loss
        loss.backward()            
        # 3. Update weights 
        optim.step()

       # E. Training Progress report for sanity purposes! 
        if verbose:
            if idx % printInterval == 0:
                    print("Epoch: {}, Batch: {} / {} ({:.0f}%). Loss: {:.4f}".format(epoch, idx,len(indata),100.*idx/len(indata), loss.item()))
                
        if stopIdx and idx == stopIdx:
                print("Stop index reached ({}). Stopping training". format(stopIdx))
                break
        
    totalSize += len(indata.dataset)
    # --|
    # F. Logging part
    avgLoss  = totalLoss  / totalSize
    model.metric = avgLoss
    # Log the current train loss
    model.history[cidx.trainLoss].append(avgLoss)
    model.history[cidx.trainAcc].append(0)
    model.history[cidx.trainAcc5].append(0)
    return loss, 1

# -----------------------------------------------------------------------------------------------------------------------

# Testing and error reports are done here
def test_classifier(model, testLoader, device, lossFn, **kwargs):
    """ DESCRIPTION: This function handles testing performance of a model. It can be modified to accept a varying number of inputs.
        
        RETURNS: acc (float): The reported average top-1 accuracy for this trial.
                 loss (float): The reported average loss on the test data, for this trial.
    """
    if 'testerArgs' in kwargs.keys():
        earlyStopIdx = kwargs['testerArgs']['earlyStopIdx'] if 'earlyStopIdx' in kwargs['testerArgs'].keys() else 0
            
    print("In Testing Function!")        
    loss = 0 
    true = 0
    acc  = 0
    # Inform Pytorch that keeping track of gradients is not required in
    # testing phase.
    with torch.no_grad():
        for idx, (data, label) in enumerate(testLoader):
            data, label = data.to(device), label.to(device)
            # output = self.forward(data)
            output = model.encoder.forward(data)
            # Sum all loss terms and tern then into a numpy number for late use.
            loss  += lossFn(output, label).item()
            # Find the max along a row but maitain the original dimenions.
            # in this case  a 10 -dimensional array.
            pred   = output.max(dim = 1, keepdim = True)
            # Select the indexes of the prediction maxes.
            # Reshape the output vector in the same form of the label one, so they 
            # can be compared directly; from batchsize x 10 to batchsize. Compare
            # predictions with label;  1 indicates equality. Sum the correct ones
            # and turn them to numpy number. In this case the idx of the maximum 
            # prediciton coincides with the label as we are predicting numbers 0-9.
            # So the indx of the max output of the network is essentially the predicted
            # label (number).
            true  += label.eq(pred[1].view_as(label)).sum().item()
            
            if earlyStopIdx and idx == earlyStopIdx:
                print("Stop index reached ({}). Stopping training". format(earlyStopIdx))
                break
                
    acc = true/len(testLoader.dataset)
    model.history[cidx.testAcc].append(acc)
    model.history[cidx.testAcc5].append(acc) 
    model.history[cidx.testLoss].append(loss) 
    # Print accuracy report!
    print("Accuracy: {} ({} / {})".format(acc, true,
                                          len(testLoader.dataset)))
    return acc, loss

# -----------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------_
def train_autoencoder(model, inData, device, lossFn, optim, **kwargs):
    ''' Description: This function evaluaes the performance of a clasifier on a provided dataset. Top 1 and
        TopTop 5 accuracy measures are provided. It handle a list of dataloaders as input.
        Arguments:  model:
                    dataLoadArgs: device: (string) Selector where the eval process will happen. cpu or cuda.
                          batchSize: (int) Size of the minibatch. Not used.
                          dataLabel: (list of string) String that acts as a label for the current dataset under
                          review. Used for printouts.
                    trainerArgs: classMethod: How the classification is done. 2 options.
                        1. Regular -> normal classification where hte utput of the network is the estimated class.
                        Usually softmax is the last layer.
                        2. Distance-> classification based on distance/dissimilarity. This is used in the siamese
                        setting. Given two encoding instances, a distance between the encoding is computed, based
                        on some space, using some measure i.e Euclidean distance or Cosine. Then, the candidate is
                        assigned the class of the sample that has the least distance. This take more time than
                        label classification.
                                printInterval: (int) The period with which priouts to the screen, of the training
                                process, occur.
                                epoch: THe current epoch. For printouts.
                    inData: (list of dataloaders). List of dataloaders  containing the datasets the
                    classiclassifier is trained on.
                    lossFn: (nn.loss module) Loss function to be used. This might be less relevant for a
                    classifier than a regressor.
    '''
    verbose = True if 'verbose' not in kwargs.keys() else kwargs['verbose']
    epoch = kwargs['trainerArgs']['epoch'] if 'trainerArgs' in kwargs.keys() else -1
    stopIdx = kwargs['trainerArgs']['stopIdx'] if 'trainerArgs' in kwargs.keys() else 0 #used for early stopping at the target batch number
    printInterval = kwargs['trainerArgs']['printInterval'] if 'trainerArgs' in kwargs.keys() else 40
    recErrorFunc = kwargs['recErrorFunc'] if 'recErrorFunc' in kwargs.keys() else utils.MSEReconstructionLoss()
    factor = 0
    totalSize, totalLoss = 0, 0
    print("Train Device is {}".format(device))
    
    # e = 0.001     # used for avoiding division by 0.
    totalSize  = 0 # used for averaging error in the end
    totalLoss  = 0 # used for logging the total loss of this epoch
    totalRecErr= 0 # used for logging the total reconstruction Error
    factor = 0    # used for division
    losses = []   # used for logging each epoch loss tensor or average loss
    
    if not isinstance(inData, list):
        inData = [inData]

    print("---------------------------------------------------------")
    print("Epoch {} Training Start\n**********************".format(epoch))
    if 'LSTM' in model.encoder.descr:
        model.encoder.embedding_net.reset_ctx()
    for setID, dSet in enumerate(inData):
        for idx, items in enumerate(dSet):
             
            # A. Forward computation input retrival and handling
            if type(items) == list:
                data   = items[0]
                target = items[1]
            else:
                target = items['target'].float() if any(items['target']) else None
                data = items['data']
            if idx == 0:
                batchSize = data.shape[0]
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)
                
            
                
            if 'LSTM' in model.encoder.descr:
                if data[0].shape[0] != batchSize:
                    continue
            # B. Forward pass calculate output of model
            # ====
            output = model.encoder.forward(*data)
            # ====|

            # C. Loss computation part.
            # Convention for all loss and reconstruction inputs is Data, Target, miscInputs. Model forward MUST be
            # Designed to match its output to the loss functions' input pattern.
            if type(output) not in (tuple, list):
                output = (output,)
            # 1 position: data
            lossInputs = (output[0],)
            recInputs  = (output[1],) + data
            # 2: positions-> rest of required misc Inputs to loss func.
            lossInputs += tuple(output[1:])

            # iac.breakpoint(True)
            # Feed tuples to loss function.
            lossOutputs = lossFn(*lossInputs)
            recOutputs = recErrorFunc(*recInputs)
            # Usually this is average loss
            loss  = lossOutputs[0] if type(lossOutputs) in (tuple, list) else lossOutputs
            recErr= recOutputs[0] if type(recOutputs) in (tuple, list) else recOutputs
            bSize = len(data[0])
            losses.append(loss.item())
            totalLoss += loss.item()
            totalRecErr += recErr.item()
            # ---|

            # D. Backpropagation part
            # 1. Zero out Grads
            optim.zero_grad()
            # 2. Perform the backpropagation based on loss
            loss.backward()
            # 3. Update weights
            optim.step()
            # ---|
            if 'LSTM' in model.encoder.descr:
                model.encoder.embedding_net.repack_ctx()
            # E. Training Progress report for sanity purposes!
            if idx % printInterval== 0:
                print("Epoch: {}-> Batch: {} / {}({:.0f}%). Batch Loss = {:.4f} Total Average Loss {:.4f} Avg Reconstruction Error {:.4f}"
                      .format(epoch, idx, len(dSet),100*(idx/len(dSet)), loss.item(), totalLoss/(idx+1), totalRecErr/(idx+1)) )
            if stopIdx and idx == stopIdx:
                print("Stop index reached ({}). Stopping training". format(stopIdx))
                break
            factor += bSize
        print("---------------------------------------------------------")

        totalSize += len(dSet.dataset)
    # --|
    # F. Logging part
    avgLoss  = totalLoss  / factor
    avgRecErr= totalRecErr/ factor
    model.metric = avgLoss
    # Log the current train loss and error
    model.history[eidx.trainLoss].append(avgLoss)   #get only the loss value
    model.history[eidx.trainRecErr].append(avgRecErr)

    return avgLoss, totalLoss
#--------------------------------------------------------------------------------------
# Start of test autoencoder.


def test_autoencoder(model, testLoaders, device, lossFn,
                     testerArgs=dict(recErrorFunc = utils.MSEReconstructionLoss(), trainMode= False, epoch=1,
                                     earlyStopIdx=0, printInterval = 10 ),
                     dataLabels = 'InputData',
                     verbose = True, **kwargs):
    ''' Description: This function evaluates the performance of a clasifier on a provided dataset. Top 1 and
        TopTop 5 accuracy measures are provided. It handles a list of dataloaders as input.
        Arguments:  model:
                    dataLabel: (list of string) String that acts as a label for the current dataset under
                                  review. Used for printouts.
                    testerArgs: recErrorFunc: Function used to get the encoders reconstruction error.
                                trainMode: Select if the function is called to perform on an evaluation
                                    setset during training, or for predictions, post-training. Mainly used for
                                    logging purposes.
                                epoch: THe current epoch. For printouts.
                                short: A selector that also doubles as the maximum batches this function
                                will consider before exiting. Used to shorten the testing procedure.
                                Usually used when the testing takes too long on a full dataset, but a
                                quick sneak-peek of the performance is required, in order to fine tune
                                traitraining.
                    testLoaders: (list of dataloaders). List of dataloaders  containing the datasets the
                    autoencoder is evaluated on.
                    lossFn: (nn.loss module) Loss function to be used.     '''

    rec = 0
    loss = 0
    totalSize = 0
    count = 0
    earlyStopIdx  = testerArgs['earlyStopIdx'] if 'earlyStopIdx' in testerArgs.keys() else 0
    epoch         = testerArgs['epoch'] if 'epoch' in testerArgs.keys() else -1
    printInterval = testerArgs['printInterval'] if 'printInterval' in testerArgs.keys() else 40
    recErrorFunc  = testerArgs['recErrorFunc']  if 'recErrorFunc' in testerArgs.keys() else utils.MSEReconstructionLoss()
    trainMode = testerArgs['trainMode'] if 'trainMode' in testerArgs.keys() else True
    verbose = True if 'verbose' not in kwargs.keys() else kwargs['verbose']
    totalLoss, totalError = 0, 0
    predictions, losses = [], []

    # Initialize parameters
    # For distance we need smallest distance from all representatives and its a scalar. AS such, we reduce on
    # y.
    # reductionAxis = 0 if (classMethod == 'distance' or classMethod == 'similarity') else 1

    # Inform Pytorch that keeping track of gradients is not required in
    # testing phase.
    with torch.no_grad():
        if not isinstance(testLoaders, list):
            testLoaders = [testLoaders]
        print("Testing on Epoch's {} model\n---".format(epoch))
        # If multiple dataloaders are given, iterate over all of them.
        for setID, dSet in enumerate(testLoaders):
            for batch_idx, items in enumerate(dSet):
                # A. Forward computation input retrival and handling
                if type(items) == list:
                    data   = items[0]
                    target = items[1]
                else:
                    target = items['target']
                    data = items['data']
                # Wrapp everything in a list or tuple. Makes handling of variable length outputs easy.
                if not type(data) in (tuple, list):
                    data = (data,)
                data = tuple(d.to(device) for d in data)
                if target is not None:
                    target = target.to(device)
                # Increase the count of seen candidates. Do not move from here.
                # Must equal to batchsize
                bSize = len(data[0])
                count += bSize
                output = model.encoder.forward(*data)
                # Update weightholder
                model.update_weight_holder(output, target)

                if type(output) not in (tuple, list):
                    output = (output,)
                # 1 position: data
                lossInputs = (output[0],)
                recInputs  = (output[1],) + data
                # 2: positions-> rest of required misc Inputs to loss func.
                lossInputs += tuple(output[1:])

                # iac.breakpoint(True)
                # Feed tuples to loss function.
                lossOutputs = lossFn(*lossInputs)
                recOutputs = recErrorFunc(*recInputs)
                # Usually this is average loss
                loss    = lossOutputs[0] if type(lossOutputs) in (tuple, list) else lossOutputs
                recError= recOutputs[0]  if type(recOutputs) in (tuple, list) else recOutputs
                losses.append(loss.item())
                totalLoss += loss.item()
                totalError+= recError.item()

                # iac.breakpoint(True)
                if batch_idx % printInterval == 0:
                    print("Batch: {} / {} ({:.0f}%). Reconstruction Error : {:.4f}, Loss: {:.4f}"
                          .format(batch_idx,len(dSet),100.*batch_idx/len(dSet), recError, loss.item()))
                # Some applications and classification datasets take too long to classify with the
                # distance approach. Use this part to classify a subset for training troubleshooting.
                if earlyStopIdx and batch_idx == earlyStopIdx:
                    print('Batch {}. Exiting early testing.'.format(earlyStopIdx))
                    break
            totalSize += len(dSet.dataset)
    # Log the current train loss
    loss = totalLoss / len(dSet)
    rec  = totalError / len(dSet)

    # Print Autoencoder's evaluation report!
    if trainMode == True:
        print("--Epoch {}  Testing --".format(epoch))
        model.history[eidx.testLoss].append(loss)   #get only the loss value
        model.history[eidx.testRecErr].append(rec)
    # This is used for prediction logging purposes.i.e logging the performance on any non-evaluation set.
    if trainMode == False:
        print("Evaluation on {}".format(dataLabels))
        model.predHistory[eidx.trainLoss].append(0)      # fill with 0, for plotting compatibility
        model.predHistory[eidx.trainRecErr].append(0)
        model.predHistory[eidx.predLoss].append(loss)   #get only the loss value
        model.predHistory[eidx.predRecErr].append(rec)

    print("Model: {}".format(model.info))
    print("Average Rec. Error: {:.4f}, Agv Loss: {:.4f}".format(totalError/len(dSet), totalLoss/len(dSet)))
    print("-------")

    return predictions, loss

# End of test Autoencoder
# ---------------------------------------------------------------------------------------_

def dynamic_conv_check(lossHistory, args = dict(window = 2, percent_change = 0.01, counter = 0, lossIdx =
                                                cidx.testLoss )):
    ''' Description: This function checks whether train should stop according to progress made.
                     If the criteria is met, training will halt.
        Arguments:   lossHistory (list): a list of lists containing loss and MAE, MAPE, as
                                         indexex by ridx file.
                     args:  A dictionary with operational parameters.
                     w:     Length of history to tbe considered
    '''
    w = args['window']
    perc = args['percent_change']
    counter = args['counter']
    lossIdx = args['lossIdx']
    # Sanitazation
    # check for parameter validity
    w = w if len(lossHistory[lossIdx]) >= w else len(lossHistory[lossIdx])
    # Increase the watch counter. When the track counter surpasses the target window length
    # check to see if progress  has been made. Essentially, by setting counter to 0 the function
    # will wait for window checks, again, before checking for convergence.
    counter += 1
    if counter >= w:
        # Compute average loss difference from history
        loss_i_1 = lossHistory[lossIdx][-w]
        avg_loss = loss_i_1
        loss_diff= 0
        for i in lossHistory[lossIdx][-w+1:]:
            loss_diff += loss_i_1 - i
            avg_loss += i
            loss_i_1 = i
        loss_diff /=w
        avg_loss /= w
        target = avg_loss * perc
        # if average loss change is less than a percentage of the average loss, exit.
        if loss_diff  < target and len(lossHistory[lossIdx]) > args['window']:
            print('!!!!!!!!!!!!!!!!!!!!')
            print("Progress over last {} epochs is less than {}% ({:.4f}<{:.4f}). Exiting Training".format(w,perc*100, loss_diff, target))
            print('!!!!!!!!!!!!!!!!!!!!')
            return 1
        else:
            return 0

    else:
        return 0
    
    
    
 