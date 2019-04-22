"""
Creates NODE model based on torchdiffeq for project document classification
Many funcitons and structures heavily adapted from rtqichen/torchdiffeq github project
https://github.com/rtqichen/torchdiffeq
"""



import pandas as pd
import numpy as np
import pickle
import time

# Import right packages
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchdiffeq import odeint_adjoint

import sys
sys.path.append("..")
import config   # Read in hyperparameters from config so that we can control for input size in networks later
import data_util
import os

# Sets to GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


tol = 1e-4

# Encodes y values into onehot vector
def onehot_encoder(y, num_classes):
	# Intialize classes
	onehot_vector = np.zeros(num_classes)

	# Assign appropriate class as 1
	onehot_vector[y] = 1

	return onehot_vector


class ConcatLinear(nn.Module):
    """
    Creates the new type hidden layer that leverages NODE solver
    """

    def __init__(self, hidden_size):
        super(ConcatLinear, self).__init__()
        module = nn.Linear
        self._layer = module(hidden_size + 1, hidden_size)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        l = self._layer(ttx)
        return l


class ODEFunc(nn.Module):
    """
    Class that holds NODE layer structure
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        
        self.l1 = ConcatLinear(hidden_size)
        self.l2 = ConcatLinear(hidden_size)

    def forward(self, t, x):
        """
        Takes in integration time argument (t) in addition to data (x)
        """

        out = F.relu(self.l1(t, x))
        out = F.sigmoid(self.l2(t, out))

        return out


        

class PLISWORK_ODE(nn.Module): # adapted from rtqichen
    def __init__(self, hidden_size):
        
        super(PLISWORK_ODE, self).__init__()
        self.odefunc = ODEFunc(hidden_size)
        self.integration_time = torch.tensor([0, 1]).float()

        
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)

        # Actual ODE solver used in forward propogation
        out = odeint_adjoint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)

        return out[1]


class FFN:
    
    def __init__(self,hidden_size, numClasses,lr,epochs):
        '''
        Creates NODE Architecture
        
        hidden_size: number of nodes in each hidden layer (int)
        numClasses: number of classes for output (int)
        lr: learning rate (int)
        epochs: number of epochs (int)
        '''
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size

        # Creates input layer for model
        input_layer = nn.Linear(config.DIM_EMBEDDING, hidden_size)
        # Creates two hidden layers with ODE solver
        hidden_layers = PLISWORK_ODE(hidden_size)
        # Creates output layer from hidden layers
        output_layer = nn.Linear(hidden_size, numClasses)
        # Strings models togeather in fully connected sequence
        self.model = nn.Sequential(input_layer, hidden_layers, output_layer).to(device)


    def fit(self,name,save_weights=False):
        '''
        Trains model using predefined number of epochs, learning rate and number of neurons in
        each hidden layer. Saves epoch results to a file name.csv, where name is replaced with the
        value put in for the name parameter.
        
        name: (str) The name of the file to save the results of this run
        save_weights: (bool) Saves the weights of the best iteration based on validation accuracy
        '''

        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        loss = nn.CrossEntropyLoss() 
        lossVal = []
        bestValAcc=0
        for i in range(self.epochs):
            start = time.time()
            loader,iteration = data_util.load_data()
            data_iter = data_util.inf_generator(loader)
            train_size = int(iteration*0.8)
            val_size = int(iteration*0.2)
            epoch_train_loss = []
            epoch_val_loss = []
            train_correct = 0
            val_correct = 0

            c = 0
            for j in range(train_size): 
                X, y = data_iter.__next__()

                X=[x.numpy()[0] for x in X] 
    
                X = Variable(torch.FloatTensor([X]), requires_grad=True).to(device)
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)

                optimizer.zero_grad()

                y_pred = self.model(X)
                output = loss(y_pred, y)
                epoch_train_loss.append(output.cpu().detach().numpy())

                if y_pred.max(-1)[1]==y:
                    train_correct += 1

                output.backward()
                optimizer.step()

                if not c % 1000:
                	print('train time:',(time.time()-start) / 60,'min', c, 'loops')
                c+=1
            
            for k in range(val_size):
                X, y = data_iter.__next__()
            
                X=[x.numpy()[0] for x in X]
    
                X = Variable(torch.FloatTensor([X]), requires_grad=True).to(device) # have to convert to tensor
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
                optimizer.zero_grad()
                y_pred = self.model(X)
                output = loss(y_pred, y)
                epoch_val_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y:
                    val_correct += 1
                valAcc = val_correct/val_size
            if save_weights and valAcc>bestValAcc:
                torch.save(self.model.state_dict(), f'../model_weights/{name}.pt') # save if we do better than current best

            end = time.time()
            lossVal.append([(end-start)/60,np.mean(epoch_train_loss),np.mean(epoch_val_loss),train_correct/train_size,val_correct/val_size]) # save values for reporting
            print('epoch time:',(end-start) / 60,'min','epoch:','{0}/{1}'.format(i,self.epochs),'train accuracy:',train_correct/train_size,', val accuracy:',val_correct/val_size)
            print(f'Train loss: {np.mean(epoch_train_loss)}		Val loss: {np.mean(epoch_val_loss)}')
        if 'model_train_results' not in os.listdir('../'):
            os.mkdir('../model_train_results')
            
        pd.DataFrame(lossVal,columns=['epoch_time','mean_train_loss','mean_val_loss','train_acc','val_acc']).to_csv('../model_train_results/'+name+'.csv',index=False) # add epoch length


    def score(self):
        loader, iteration = data_util.load_data(partition='test')
        
        data_iter = data_util.inf_generator(loader)
        correct = 0
        for i in range(iteration):
            X, y = data_iter.__next__()
            
            X=[x.numpy()[0] for x in X] 
            
            predX = Variable(torch.FloatTensor([X]), requires_grad=True).to(device)
            y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
            
            y_pred = self.model(predX)
            
            if y_pred.max(-1)[1]==y:
                correct += 1
        return correct/iteration

    def test(self,name):
        '''
        Returns the top-1 accuracy on an unseen test dataset.
        '''

        loader, iteration = data_util.load_data(partition='test')
        
        data_iter = data_util.inf_generator(loader)
        results = []
        for i in range(iteration):
            X, y = data_iter.__next__()
            
            X=[x.numpy()[0] for x in X] 
            
            predX = Variable(torch.FloatTensor([X]), requires_grad=True).to(device)
            y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
            
            y_pred = self.model(predX)
            
            results.append([y.cpu().numpy()[0],y_pred.max(-1)[1].cpu().numpy()[0]])
        pd.DataFrame(results,columns=['y_true','y_pred']).to_csv(f'../model_train_results/{name}.csv',index=False)



if __name__=='__main__':
    hiddenSize = [200]
    for i in range(len(hiddenSize)):
        print(f'-------starting grid search {i}----------')
        print(hiddenSize[i])
        model = FFN(hidden_size = hiddenSize[i],
                      numClasses=20,
                      lr=0.01,
                      epochs=15)
        print('start')
        model.fit(name=f'node_200',save_weights=True)
        print(model.score())
        model.test(name='node_200_test')
