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


class ShallowCNN(nn.Module):
    
    def __init__(self,input_size,kern1,kern2,kern3,num_filters,num_fc,num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=num_filters,kernel_size=kern1)
        self.conv2 = nn.Conv1d(in_channels=1,out_channels=num_filters,kernel_size=kern2)
        self.conv3 = nn.Conv1d(in_channels=1,out_channels=num_filters,kernel_size=kern3)
        
        self.fc1 = nn.Linear(in_features = num_filters, out_features = num_fc)
        self.fc2 = nn.Linear(in_features = num_fc, out_features=num_classes)
    
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        x1 = F.max_pool1d(x1)
        x2 = F.max_pool1d(x2)
        x3 = F.max_pool3d(x3)
        
        return F.log_softmax(self.fc2(self.fc1(torch.cat(x1,x2,x3))))
        
        
class ShallowCNNModel:
    def __init__(self,batch_size,lr,epochs,input_size,kern1,kern2,kern3,num_filters,num_fc):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs=epochs
        self.model = ShallowCNN(input_size,kern1,kern2,kern3,num_filters,num_fc).cuda()
    def fit(self,X,y):
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.4)
        loss = nn.CrossEntropyLoss() # cross-entropy loss
        for i in range(self.epochs):
            X_batch = X.sample(n=self.batch_size)
            y_batch = y[X_batch.index]

            inputX = Variable(torch.FloatTensor([X_batch.values]), requires_grad=True).cuda() # have to convert to tensor
            inputY = Variable(torch.FloatTensor([y_batch.values]), requires_grad=False).cuda()
            optimizer.zero_grad()
            y_pred = self.model(inputX)
            output = loss(y_pred, inputY)

            output.backward()
            optimizer.step()
    def predict(self,X):
        predX = Variable(torch.FloatTensor(X.values)).cuda()
        proba = self.model(predX)

        return proba

if __name__=='__main__':
    DATALOC = 'C:/Users/ibiyt/Desktop/GitHub/CS6120-Term-Project/data'
    df = pd.read_pickle(DATALOC+'/'+'newsgroup_vectors.pkl')