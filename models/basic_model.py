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
    '''
    Feed-forward neural network blocks. Created this class in order to inter-
    change these blocks and ODE blocks.
    '''
    def __init__(self, hidden_size):
        super(PLISWORK, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size) # hidden layers
        self.l2 = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)

        
        return out



class FFN:
    
    def __init__(self,hidden_size, numClasses,lr,epochs):
        '''
        Creates FFN architecture
        
        hidden_size: (int) number of neurons per hidden layer
        numClasses: (int) number of classes
        lr: (float) Learning rate
        epochs: (int) number of iterations the model learns on.
        '''
        self.lr = lr
        self.epochs = epochs
        
        input_layer = nn.Linear(config.DIM_EMBEDDING, hidden_size)
        hidden_layers = PLISWORK(hidden_size)
        output_layer = nn.Linear(hidden_size, numClasses)
        self.model = nn.Sequential(input_layer, hidden_layers, output_layer).to(device)


    def fit(self,name,save_weights=False):
        '''
        Trains model using predefined number of epochs, learning rate and number of neurons in
        each hidden layer. Saves epoch results to a file name.csv, where name is replaced with the
        value put in for the name parameter.
        
        name: (str) The name of the file to save the results of this run
        save_weights: (bool) Saves the weights of the best iteration based on validation accuracy
        '''
        print(name)
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        loss = nn.CrossEntropyLoss() # cross-entropy loss
        lossVal = [] # to create dataframe for logging
        bestValAcc=0 # to check for saving weights
        for i in range(self.epochs):
            start = time.time()
            loader,iteration = data_util.load_data()
            data_iter = data_util.inf_generator(loader) # inf generator taken from rtiqchen implementation of NODE
            train_size = int(iteration*0.8) # takes 80% of data as train and 20 as test
            val_size = int(iteration*0.2)
            epoch_train_loss = [] # collect values to update log for post-processing
            epoch_val_loss = []
            train_correct = 0
            val_correct = 0

            for j in range(train_size): #calculated train size to do train dev split will calculate mean loss at end
                X, y = data_iter.__next__()

                X=[x.numpy()[0] for x in X] 
                
    
                X = Variable(torch.FloatTensor([X]), requires_grad=True).to(device) # have to convert to tensor
                #print(X.shape)
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)

                optimizer.zero_grad()
                y_pred = self.model(X)
                output = loss(y_pred, y)
                epoch_train_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y: # if max log softmax corresponds to correct class add 1
                    train_correct += 1

                output.backward()
                optimizer.step()

                #print(list(self.model.parameters()))
            
            for k in range(val_size):
                X, y = data_iter.__next__()
            
                X=[x.numpy()[0] for x in X]
    
                X = Variable(torch.FloatTensor([X]), requires_grad=True).to(device) # have to convert to tensor
    
                y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
                optimizer.zero_grad()
                y_pred = self.model(X)
                output = loss(y_pred, y)
                epoch_val_loss.append(output.cpu().detach().numpy())
                if y_pred.max(-1)[1]==y: # if max log softmax corresponds to correct class add 1
                    val_correct += 1
                valAcc = val_correct/val_size
            if save_weights and valAcc>bestValAcc:
                torch.save(self.model.state_dict(), f'../model_weights/{name}.pt') # save if we do better than current best

            end = time.time()
            lossVal.append([(end-start)/60,np.mean(epoch_train_loss),np.mean(epoch_val_loss),train_correct/train_size,val_correct/val_size]) # save values for reporting
            print('epoch time:',(end-start) / 60,'min','epoch:','{0}/{1}'.format(i,self.epochs),'train accuracy:',train_correct/train_size,', val accuracy:',val_correct/val_size)
            print(f'Train loss: {np.mean(epoch_train_loss)}     Val loss: {np.mean(epoch_val_loss)}')
        if 'model_train_results' not in os.listdir('../'):
            os.mkdir('../model_train_results')
            
        pd.DataFrame(lossVal,columns=['epoch_time','mean_train_loss','mean_val_loss','train_acc','val_acc']).to_csv('../model_train_results/'+name+'.csv',index=False) # add epoch length


    def score(self):
        '''
        Returns the top-1 accuracy on an unseen test dataset.
        '''
        loader, iteration = data_util.load_data(partition='test')
        #iteration = 1
        data_iter = data_util.inf_generator(loader)
        correct = 0
        for i in range(iteration):
            X, y = data_iter.__next__()
            
            X=[x.numpy()[0] for x in X] 
            
            predX = Variable(torch.FloatTensor([X]), requires_grad=True).to(device)
            y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
            
            y_pred = self.model(predX)
            
            if y_pred.max(-1)[1]==y: # if class corresponding to max log softmax is the ground truth class 
                correct += 1
        return correct/iteration

    def test(self,name):
        '''
        Takes a supposedly unseen dataset and finds the top-1 predicted labels.
        Stores those values in a csv file called name.csv, where name is the value
        set for the name parameter for this function.
        '''
        loader, iteration = data_util.load_data(partition='test')
        #iteration = 1
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
    #filters,fc_inputs,poolStride,blockFunc,blockSize,kmax,numClasses,lr,epochs
    hiddenSize = [50,75,100,200]

    for i in range(len(hiddenSize)):
        print(f'-------starting grid search {i}----------')
        model = FFN(hidden_size = hiddenSize[i],
                    numClasses=20,
                    lr=0.01,
                    epochs=15)
        print('start')
        model.fit(name=f'default_run{i}')
        print(model.score())
    
    # best model by default was 200 neurons
    model = FFN(hidden_size = 200,
                  numClasses=20,
                  lr=0.01,
                  epochs=15)
    print('start')
    model.fit(name='ffn_200')
    model.test(name='ffn_200_test')