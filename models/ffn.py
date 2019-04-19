# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:36:01 2019

@author: ibiyt
"""

import pandas as pd
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import time



import sys
sys.path.append("..")
import data_util
import os
import warnings
import config
warnings.filterwarnings("ignore") # fuck warnings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FFN(nn.Module):
    '''
    Extremely basic feed forward neural network with the following structure
    input -> linear -> ReLU -> linear -> ReLU -> softmax
    Trained using stochastic gradient descent, batch size = 1
    '''
    
    def __init__(self,input_size,hidden1,hidden2,num_classes):
        super().__init__()
        self.inputLayer = nn.Linear(input_size,hidden1)
        self.hiddenLayer1 = nn.Linear(hidden1,hidden2)
        self.hiddenLayer2 = nn.Linear(hidden2,num_classes)
    
    def forward(self,x):
        
        x = self.inputLayer(x)
        x = F.relu(self.hiddenLayer1(x))
        x = F.relu(self.hiddenLayer2(x))
        

        return F.log_softmax(x)

class FNNModel:
    
    def __init__(self,epochs,batch_size,lr,input_size,hidden1,hidden2,num_classes):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = FFN(input_size,hidden1,hidden2,num_classes).to(device)
    
    def fit(self,name):
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.4)
        loss = nn.CrossEntropyLoss() # cross-entropy loss
        lossVal = []
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
            for j in range(1): #calculated train size to do train dev split will calculate mean loss at end
                X, y = data_iter.__next__()
            
                X=[x.numpy()[0] for x in X]
                x_means = np.mean(np.array(X), axis = 0)
                
    
                X = Variable(torch.FloatTensor([x_means]), requires_grad=True).to(device)

                
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)

                optimizer.zero_grad()
                y_pred = self.model(X)
                #print(y_pred.max(-1)[1])
                output = loss(y_pred, y)
                epoch_train_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y:
                    train_correct += 1

                output.backward()
                optimizer.step()
                print(list(self.model.parameters()))
            for k in range(1):
                X, y = data_iter.__next__()
            
                X=[x.numpy()[0] for x in X] 
                x_means = np.mean(np.array(X), axis = 0)
    
                X = Variable(torch.FloatTensor([x_means]), requires_grad=True).to(device) # have to convert to tensor
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
                optimizer.zero_grad()
                y_pred = self.model(X)

                output = loss(y_pred, y)

                epoch_val_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y:
                    val_correct += 1

            lossVal.append([np.mean(epoch_train_loss),np.mean(epoch_val_loss),train_correct/train_size,val_correct/val_size])
            print('epoch time:',time.time()-start,'seconds','epoch:','{0}/{1}'.format(i,self.epochs),'train accuracy:',train_correct/1,', val accuracy:',val_correct/val_size)
        if 'model_train_results' not in os.listdir('../'):
            os.mkdir('../model_train_results')
            
        pd.DataFrame(lossVal,columns=['mean_train_loss','mean_val_loss','train_acc','val_acc']).to_csv('../model_train_results/'+name+'.csv',index=False)



        torch.save(self.model.state_dict(), '../model_weights/shallowCNN.pt')

if __name__=='__main__':
    model = FNNModel(epochs=15,batch_size=config.BATCH_SIZE,lr=0.011,input_size=config.DIM_EMBEDDING,hidden1=10,hidden2=10,num_classes=20)
    print('start')
    model.fit(name='default_run')     
        