# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:45:24 2019

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

# Sets to GPU if possible
# Might be wrong device name for gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

class ShallowCNN(nn.Module):
    
    def __init__(self,emb_size,kern1,kern2,kern3,num_filters,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=[kern1,emb_size])
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=[kern2,emb_size])
        self.conv3 = nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=[kern3,emb_size])
          
        self.fc1 = nn.Linear(in_features = num_filters*3, out_features = num_classes)
   
    
    def forward(self,x):
        x=x.unsqueeze(1)

        x1 = torch.squeeze(F.relu(self.conv1(x)),-1)
        x2 = torch.squeeze(F.relu(self.conv2(x)),-1)
        x3 = torch.squeeze(F.relu(self.conv3(x)),-1)
        

        
        x1 = F.max_pool1d(x1,x1.size(2))
        x2 = F.max_pool1d(x2,x2.size(2))
        x3 = F.max_pool1d(x3,x3.size(2))


        out = torch.cat([x1,x2,x3],2) # shape is (input_size*3,num_filters)
        
        out = out.view(out.size(0),-1)
        
        out = self.fc1(out)
   
        return F.log_softmax(out)
        
        
class ShallowCNNModel:
    def __init__(self,emb_size,batch_size,lr,epochs,kern1,kern2,kern3,num_filters,num_classes):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs=epochs
        self.model = ShallowCNN(emb_size,kern1,kern2,kern3,num_filters,num_classes).to(device)

        if config.LOAD_MODEL:
            self.model.load_state_dict(torch.load('../model_weights/shallowCNN.pt'))

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

            
            for j in range(train_size): #calculated train size to do train dev split will calculate mean loss at end
                X, y = data_iter.__next__()
            
                X=[x.numpy()[0] for x in X] 
    
                X = Variable(torch.FloatTensor([X]), requires_grad=True).to(device) # have to convert to tensor
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
                optimizer.zero_grad()
                y_pred = self.model(X)
                output = loss(y_pred, y)
                epoch_train_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y:
                    train_correct += 1

                output.backward()
                optimizer.step()
            
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

            lossVal.append([np.mean(epoch_train_loss),np.mean(epoch_val_loss),train_correct/train_size,val_correct/val_size])
            print('epoch time:',time.time()-start,'seconds','epoch:','{0}/{1}'.format(i,self.epochs),'train accuracy:',train_correct/train_size,', val accuracy:',val_correct/val_size)
        if 'model_train_results' not in os.listdir('../'):
            os.mkdir('../model_train_results')
            
        pd.DataFrame(lossVal,columns=['mean_train_loss','mean_val_loss','train_acc','val_acc']).to_csv('../model_train_results/'+name+'.csv',index=False)



        torch.save(self.model.state_dict(), '../model_weights/shallowCNN.pt')

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

        

if __name__=='__main__':
    model = ShallowCNNModel(100,config.BATCH_SIZE,0.001,80,1,2,3,80,20)
    print('start')
    model.fit(name='default_run')
    print(model.score())
