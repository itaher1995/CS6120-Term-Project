"""
Creates NODE model based on torchdiffeq for project document classification
"""

# Import right packages
from torchdiffeq import odeint_adjoint
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
# Get batch data

# Create ODE network based on pytorch objects in ODEfunc class
	# Needs to have at least __init__ and forward functions
	# Where __init__ has the network declaration and forward has the forward propogation

class TemporalConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Temporal Convolution Layer (kernel size = 3, stride = 1)
        # Temporal Batch Norm
        # ReLU
        
        # Temporal Convolution Layer (kernel size = 3, stride)
        # Temporal Batch Norm
        # ReLU
        return None
    def forward(x):
        return x

class ODEFunc(nn.Module):
    
    def __init__(self,input_size,kernel_size,feature_maps,embedding_dim,num_classes,convStride,poolStride,resnet=False):
        super().__init__()
        
        # Start by getting the embedding for a specific sentence
        # Start with a temporal convolutional layer with stride 3 and feature maps, X
        
        #create n conv blocks with stride 3 and n feature maps
       
        # each conv block is two sets of temporal convolutions with size 3 and n feature map, temporal batch norm and relu
        # Use max pooling with stride 2 and size 3
        
        # m 2 convblock max pool groups
        # final 2 convblock and then k-max pool
        
        # 3 fc layers with relu
        
        self.nfe = 0 # not really sure what this is yet.
    

    def forward(self, t, x):
        self.nfe += 1
        

        
        return x

class ODEBlock(nn.Module): # adapted from rtqichen
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        
        out = odeint_adjoint(self.odefunc, x, self.integration_time)
        
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def learning_rate_with_decay(lr,batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates): # taken from rtqichen
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

        
class convNODENET:
    def __init__(self,input_size,batch_size,epochs,lr,batches_per_epoch,num_classes,dropout):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.batches_per_epoch = batches_per_epoch
        self.num_classes = num_classes
        self.dropout = dropout
        self.odeBlock = ODEBlock(ODEFunc(input_size,kernel_sizes=[1,2,3],num_kernels=100,embedding_dim=100,num_classes=num_classes,dropout=dropout))
        self.model = nn.Sequential(self.odeBlock,nn.Softmax())#.cuda()
    def fit(self,X,y):

        
        
        criterion = nn.CrossEntropyLoss()
        
        
        lr_fn = learning_rate_with_decay(
        self.lr, self.batch_size, batch_denom=128, batches_per_epoch=self.batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        for itr in range(self.epochs * self.batches_per_epoch):
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(itr)

            X_batch = X.sample(n=self.batch_size)
            y_batch = y[X_batch.index]

            inputX = Variable(torch.FloatTensor([X_batch.values]), requires_grad=True)#.cuda() # have to convert to tensor
            inputY = Variable(torch.FloatTensor([y_batch.values]), requires_grad=False)#.cuda()
            optimizer.zero_grad()
            
            logits = self.model(inputX)
            
            loss = criterion(logits, inputY)
    
            self.odeBlock[0].nfe = 0
    
            loss.backward()
            optimizer.step()
            
            self.odeBlock[0].nfe = 0
            
    def predict(self,X):
        predX = Variable(torch.FloatTensor(X.values)).cuda()
        proba = self.model(predX)
        return proba
        

# Some type of reporting class, if we're modeling after NODE peoples work

# train funciton
	# Initialize layers based one ODEfunc
	# Define loss criterion and optimizer
	# Leverage odeint (main funciton from project) to calculate prediction using time as the integral intervales (I still don't perfectly understand)

# test function
	# Used saved model to test things
    
if __name__=='__main__':
    features = pd.DataFrame([[np.random.rand() for i in range(100)] for j in range(10)])
    response = pd.Series([np.random.randint(0,3) for i in range(10)])
    convNode = convNODENET(features.shape[1],batch_size=1,epochs=10,lr=0.001,batches_per_epoch=1,num_classes=10,dropout=0)
    convNode.fit(features,response)