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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Cudacris goin in on the verse // cause I never use cpu and I won't start now

class PLISWORK(nn.Module):
    def __init__(self, hidden_size):
        super(PLISWORK, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        
        return out



class FFN:
    
    def __init__(self,hidden_size, numClasses,lr,epochs):
        '''
        Creates VDCNN Architecture
        
        ksize: kernel size (int)
        filters: list of number of filters (list)
        poolStride: stride size for pooling layers (int)
        blockFunc: TempConvBlock or ODEBlock
        '''
        self.lr = lr
        self.epochs = epochs
        
        input_layer = nn.Linear(config.DIM_EMBEDDING, hidden_size)
        hidden_layers = PLISWORK(hidden_size)
        output_layer = nn.Linear(hidden_size, numClasses)
        self.model = nn.Sequential(input_layer, hidden_layers, output_layer).to(device)


    def fit(self,name):
        
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
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
                x_means = np.mean(np.array(X), axis = 0)
    
                X = Variable(torch.FloatTensor([x_means]), requires_grad=True).to(device) # have to convert to tensor
                #print(X.shape)
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)

                optimizer.zero_grad()
                y_pred = self.model(X)
                output = loss(y_pred, y)
                epoch_train_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y:
                    train_correct += 1

                output.backward()
                optimizer.step()
                #print(list(self.model.parameters()))
            
            for k in range(val_size):
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
            print('epoch time:',(time.time()-start) / 60,'min','epoch:','{0}/{1}'.format(i,self.epochs),'train accuracy:',train_correct/train_size,', val accuracy:',val_correct/val_size)
            print(f'Train loss: {np.mean(epoch_train_loss)}		Val loss: {np.mean(epoch_val_loss)}')
        if 'model_train_results' not in os.listdir('../'):
            os.mkdir('../model_train_results')
            
        pd.DataFrame(lossVal,columns=['mean_train_loss','mean_val_loss','train_acc','val_acc']).to_csv('../model_train_results/'+name+'.csv',index=False)



        torch.save(self.model.state_dict(), '../model_weights/FFN.pt')

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
    #filters,fc_inputs,poolStride,blockFunc,blockSize,kmax,numClasses,lr,epochs
    hiddenSize = [5,10,15,20,25,50,75,100]

    for i in range(len(hiddenSize)):
        print(f'-------starting grid search {i}----------')
        model = FFN(hidden_size = hiddenSize[i],
                      numClasses=20,
                      lr=0.01,
                      epochs=50)
        print('start')
        model.fit(name=f'default_run{i}')
    #print(model.score())