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
#warnings.filterwarnings("ignore") # fuck warnings

# Sets to GPU if possible
# Might be wrong device name for gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Cudacris goin in on the verse // cause I never use cpu and I won't start now
      
def subBlock(input_size, num_filters, ksize):
    '''
    For VDCNN, each convolution block is defined by its
    '''
    return nn.Conv1d(in_channels=input_size, # iterates over each row in the input. Think as if it is iterating over each embedding
                                   out_channels=num_filters, 
                                   kernel_size=ksize),
            nn.BatchNorm1d(num_features=num_filters),
            nn.LeakyReLU(inplace=True) # Original papers use ReLU, LeakyRelU makes sure the differentiation for negative values is not zero, but rather a very low value. Helps with vanishing gradient problem
            
def TempConvBlock(input_size, num_filters, ksize):
    return [*subBlock(input_size, num_filters,ksize),
            *subBlock(num_filters, num_filters,ksize)]

def createBlock(blockFunc,input_size,num_filters,ksize,blockSize):
    '''
    Creates a VDCNN Block where blockSize is the number of TempConvBlocks in the block
    
    blockFunc: TempConvBlock or ODEBlock
    num_filters: number of filters (int)
    ksize: kernel size (int)
    blockSize: The amount of blockFuncs in this function (int)
    
    e.g. blockSize = 3
    
    return [blockFunc,blockFunc,blockFunc]
    '''

    block = [*blockFunc(input_size, num_filters,ksize)]
    
    block.extend([*blockFunc(num_filters,num_filters,ksize)]*blockSize)
    
    return block
    

def fc(num_input,num_output):
    return [nn.Linear(in_features=num_input,
                      out_features=num_output),
            nn.LeakyReLU(inplace=True)]

#def kmax_pooling(x, dim, k):
#    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
#    return x.gather(dim, index)

class kmax_pooling(nn.Module):
    def __init__(self,k):
        super(kmax_pooling,self).__init__()
        self.k = k
    def forward(self,x):
        return x.topk(self.k)[0].view(1,-1)


class PLISWORK(nn.Module):
    def __init__(self,ksize,filters,fc_inputs,poolStride,blockFunc,blockSize,kmax,numClasses):
        self.conv1d = nn.Conv1d(in_channels=config.DIM_EMBEDDING,
                                             out_channels=filters[0],
                                             kernel_size=ksize,
                                             padding=0)
        self.block1 = *createBlock(blockFunc=blockFunc,
                                                 input_size=filters[0],
                                                 num_filters=filters[0],
                                                 ksize=ksize,
                                                 blockSize=blockSize)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.block2 = *createBlock(blockFunc=blockFunc,
                                                 input_size = filters[0],
                                                 num_filters=filters[0],
                                                 ksize=ksize,
                                                 blockSize=blockSize)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.block3 = *createBlock(blockFunc=blockFunc,
                                                 input_size=filters[0],
                                                 num_filters=filters[0],
                                                 ksize=ksize,
                                                 blockSize=blockSize)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.block4 = *createBlock(blockFunc=blockFunc,
                                                 input_size=filters[0],
                                                 num_filters=filters[0],
                                                 ksize=ksize,
                                                 blockSize=blockSize)
        self.kpool = kmax_pooling(k=kmax)
        self.fc1 = *fc(kmax*filters[0],(kmax*filters[0])//2)
        self.fc2 = *fc((kmax*filters[0])//2,(kmax*filters[0])//2)
        self.lin = nn.Linear((kmax*filters[0])//2,numClasses)

    def forward(self, x)
        out = self.conv1d(x)
        out = self.block1(out)
        out = self.pool1(out)
        out = self.block2(out)
        out = self.pool2(out)
        out = self.block3(out)
        out = self.pool3(out)
        out = self.block4(out)
        out = self.kpool(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.lin(out)

        return out

    
class VDCNN:
    
    def __init__(self,ksize,filters,fc_inputs,poolStride,blockFunc,blockSize,kmax,numClasses,lr,epochs):
        '''
        Creates VDCNN Architecture
        
        ksize: kernel size (int)
        filters: list of number of filters (list)
        poolStride: stride size for pooling layers (int)
        blockFunc: TempConvBlock or ODEBlock
        '''
        self.lr = lr
        self.epochs = epochs
        self.model = PLISWORK(ksize,filters,fc_inputs,poolStride,blockFunc,blockSize,kmax,numClasses).to(device)
    
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

                X = X.squeeze().t().unsqueeze(0)
                print(X.shape)
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



        torch.save(self.model.state_dict(), '../model_weights/VDCNN.pt')

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
    model = VDCNN(ksize=3,filters=[64,128,256,512],
                  fc_inputs=[4096,2048],
                  poolStride=2,
                  blockFunc=TempConvBlock,
                  blockSize=2,
                  kmax=8,
                  numClasses=20,
                  lr=0.001,
                  epochs=1)
    print('start')
    model.fit(name='default_run')
    print(model.score())