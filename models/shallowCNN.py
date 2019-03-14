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

import sys
sys.path.append("..")
import data_util


# Sets to GPU if possible
# Might be wrong device name for gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

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
        self.model = ShallowCNN(input_size,kern1,kern2,kern3,num_filters,num_fc).to(device)

        loader = data_util.load_data()
        data_iter = data_util.inf_generator(loader)

    def fit(self,X,y):
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.4)
        loss = nn.CrossEntropyLoss() # cross-entropy loss
        for i in range(self.epochs):
            X, Y = data_iter.__next__()

            X = X.to(device) # have to convert to tensor
            Y = Y.to(device)
            optimizer.zero_grad()
            y_pred = self.model(X)
            output = loss(y_pred, Y)

            output.backward()
            optimizer.step()
    def predict(self,X):
        predX = X.to(device)
        prob = self.model(predX)

        return prob

if __name__=='__main__':
    print('Taken care of')