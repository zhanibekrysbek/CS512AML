
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from Classifier import LSTMClassifier
import matplotlib.pyplot as plt
import numpy as np
import warnings


warnings.filterwarnings("ignore")

torch.manual_seed(1)
# Hyperparameters, feel free to tune


batch_size = 27
output_size = 9   # number of class
hidden_size = 50  # LSTM output size of each time step
input_size = 12
basic_epoch = 201
Adv_epoch = 101
Prox_epoch = 101



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)



# Training model
def train_model(model, train_iter, mode, epsilon=0.01):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        inX = batch[0]
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        r = 0
        optim.zero_grad()
        prediction = model(inX, r, batch_size = inX.size()[0], mode = mode)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/(inX.size()[0])
        if mode == 'AdvLSTM':
            ''' Add adversarial training term to loss'''
            pert = compute_perturbation(loss, model)
            r =  epsilon * pert
            prediction2 = model(inX, r, batch_size = inX.size()[0], mode = 'AdvLSTM')
            loss += loss_fn(prediction2, target)
            num_corrects2 = (torch.max(prediction2, 1)[1].view(target.size()).data == target.data).float().sum()
            num_corrects += num_corrects2
            acc = 100.0 * num_corrects/(2*inX.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


# Test model
def eval_model(model, test_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    r = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            input = batch[0]
            target = batch[1]
            target = torch.autograd.Variable(target).long()
            prediction = model(input, r, batch_size=input.size()[0], mode = mode)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects.double()/(input.size()[0])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)




def compute_perturbation(loss, model):

    '''need to be implemented'''
    # Use autograd
    gradDir = grad(loss, model.inX, retain_graph = True)[0] # This returns a tuple
    magnitude = gradDir.pow(2).sum().sqrt()
    '''need to be implemented'''

    return gradDir/magnitude #the value of g / ||g||


# train_test = 0 plots train acc
# train_test = 1 plots test acc
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


#def main():




train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)
loss_fn = F.cross_entropy

print(" ==================================")
print("|     Basic model training         |")
print(" ==================================")
''' Training basic model '''
model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
history = [[],[]]

for epoch in range(basic_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
        train_loss, train_acc = train_model(model, train_iter, mode = 'plain')
        val_loss, val_acc = eval_model(model, test_iter, mode ='plain')
        if epoch%50 == 0:
            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
        history[0].append(train_acc)
        history[1].append(val_acc)
    

''' Save and Load model'''
# 1. Save the trained model from the basic LSTM
plot(history, train_test = 1,saveFile = './plots/LSTM_plain_acc_test.png')
plot(history, train_test = 0, saveFile = './plots/LSTM_plain_acc_train.png')
torch.save(model, './models/LSTM_Plain_model_cell')


print(" ==================================")
print("|     Adv_model training           |")
print(" ==================================")


# 2. load the saved model to Adv_model, which is an instance of LSTMClassifier
Adv_model = torch.load('./models/LSTM_Plain_model_cell')


''' Training Adv_model'''
epsilons = [0.01, 0.1, 1.0]
Adv_epochs = [50,50, 100]
history = []

for ind, eps in enumerate(epsilons):
    hist = [[],[]]
    print(f" ==============  Training eps: {eps} ")
    Adv_model = torch.load('./models/LSTM_Plain_model_cell')
    for epoch in range(Adv_epochs[ind]):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
        train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM', epsilon=eps)
        val_loss, val_acc = eval_model(Adv_model, test_iter, mode = 'AdvLSTM')
        hist[0].append(train_acc)
        hist[1].append(val_acc)
        if epoch % 50 == 0:
            print(f'\t epch: {epoch+1:02}, TrLoss: {train_loss:.3f}, TrAcc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
    history.append(hist)

plot(history,eps=epsilons,title="Adversarial LSTM training", figType='lrCurve',saveFile = './plots/AdvLSTM_acc_lrCurve_test.png')
plot(history,train_test=0,eps=epsilons,title="Adversarial LSTM training", figType='lrCurve',saveFile = './plots/AdvLSTM_acc_lrCurve_train.png')

torch.save(Adv_model, './models/Adv_model_cell')



print(" ==================================")
print("|     Prox_model training          |")
print(" ==================================")

# 4. load the saved model to Prox_model, which is an instance of LSTMClassifier
Prox_model = torch.load('./models/Adv_model_cell')

"""" Check the performance of Prox_model with pretrained model"""
eps = [0.1, 1.,5.]
for ep in eps:
    Prox_model.prox_eps = ep
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
    print(f"prox_eps: {Prox_model.prox_eps}, val_loss: {val_loss}, val_acc: {val_acc}")

"""
''' Training Prox_model'''
for epoch in range(Prox_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM')
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')

"""



#if __name__ == '__main__':
#    main()


