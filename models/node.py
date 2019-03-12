"""
Creates NODE model based on torchdiffeq for project document classification
"""



import pandas as pd
import numpy as np
import pickle
from time import time

# Import right packages
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchdiffeq import odeint_adjoint

# Get batch data

# Create ODE network based on pytorch objects in ODEfunc class
	# Needs to have at least __init__ and forward functions
	# Where __init__ has the network declaration and forward has the forward propogation

# Sets to GPU if possible
# Might be wrong device name for gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Loads features and puts them into tensor form
def load_features():

	print('Loading data...')

	start = time()

	# Loads Sentence vectors
	with open('data/sent_embds.pkl', 'rb') as f:
		feature_vecs = pickle.load(f)

	# Gets metadata, including source (aka Y data)
	metadata = pd.read_pickle('data/newsgroup_map.pkl')

	# Stores feature vectors in tensor
	X = torch.tensor(feature_vecs).to(device)

	# Gets source targets and stores in tensor
	Y = [[v] for v in metadata.target.tolist()]
	Y = torch.tensor(y).to(device)

	print('Data loaded in:', (time() - start) / 60, 'min')

	return X, Y
# Get batch data

# Create ODE network based on pytorch objects in ODEfunc class
	# Needs to have at least __init__ and forward functions
	# Where __init__ has the network declaration and forward has the forward propogation

class TemporalConvBlock(nn.Module):
    def __init__(self, num_inputs,num_filters,kernel_size):
        super().__init__()
        
        # Temporal Convolution Layer (kernel size = 3, stride = 1)
        self.tcl1 = nn.Conv1d(in_channels= num_inputs,out_channels= num_filters,kernel_size = kernel_size)
        # Temporal Batch Norm
        self.tbn1 = nn.BatchNorm1d(num_features=num_filters)
        # ReLU (apply in forward step)
        
        # Temporal Convolution Layer (kernel size = 3, stride)
        self.tcl2 = nn.Conv1d(in_channels=num_filters,out_channels=num_filters,kernel_size=kernel_size)
        # Temporal Batch Norm
        self.tbn2 = nn.BatchNorm1d(num_features=num_filters)
        # ReLU (apply in forward step)

    def forward(self,x):
        return F.relu(self.tbn2(self.tcl2(F.relu(self.tbn1(self.tcl1(x))))))

class ODEFunc(nn.Module):
    
    def __init__(self,input_size,numConvBlocks,kernel_size,feature_maps,embedding_dim,num_classes,resnet=False):
        super().__init__()
        
        
        # Start by getting the embedding for a specific sentence
        self.feature_maps = feature_maps
        self.numConvBlocks = numConvBlocks
        
        self.tempConv1 = nn.Conv1d(in_channels=1,out_channels=self.feature_maps[0],kernel_size=kernel_size)
        
        # Start with a temporal convolutional layer with stride 3 and feature maps, X
        #create n conv blocks with stride 3 and n feature maps
        # each conv block is two sets of temporal convolutions with size 3 and n feature map, temporal batch norm and relu
        # m 2 convblock max pool groups
        self.convBlocks1 = [TemporalConvBlock(num_inputs=self.feature_maps[0],num_filters=self.feature_maps[0],kernel_size=kernel_size) for i in range(self.numConvBlocks)]

        self.maxpool1 = nn.MaxPool1d(kernel_size=kernel_size,stride=2) # Use max pooling with stride 2 and size 3
        
        self.convBlocks2 = [TemporalConvBlock(num_inputs=self.feature_maps[0],num_filters=self.feature_maps[1],kernel_size=kernel_size) for i in range(self.numConvBlocks)]

        self.maxpool2 = nn.MaxPool1d(kernel_size=kernel_size,stride=2)
        
        self.convBlocks3 = [TemporalConvBlock(num_inputs=self.feature_maps[1],num_filters=self.feature_maps[2],kernel_size=kernel_size) for i in range(self.numConvBlocks)]
        
        self.maxpool3 = nn.MaxPool1d(kernel_size=kernel_size,stride=2)
        
        self.maxpool4 = nn.MaxPool1d(kernel_size=kernel_size,stride=2)
        
        # final 2 convblock and then k-max pool
        
        self.convBlocks4 = [TemporalConvBlock(num_inputs=self.feature_maps[2],num_filters=self.feature_maps[3],kernel_size=kernel_size) for i in range(numConvBlocks)]
        
        # 3 fc layers with relu
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,num_classes)

    def forward(self, t, x):
        
        def kmax_pooling(x, dim, k):
            index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
            return x.gather(dim, index)
        
        x = self.tempConv1(x)
        
        for i in range(self.numConvBlocks):
            x=self.convBlocks1[i](x)
        x = self.maxpool1(x)
        
        for i in range(self.numConvBlocks):
            x=self.convBlocks2[i](x)
        
        x = self.maxpool2(x)
        
        for i in range(self.numConvBlocks):
            x=self.convBlocks3[i](x)
        
        x = self.maxpool3(x)
        
        for i in range(self.numConvBlocks):
            x = self.convBlocks4[i](x)
         
        x = self.maxpool4(x)    
        #x = kmax_pooling(x, self.feature_maps[3], 8)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        
        x = self.fc2(x)
        
        x = self.fc3(x) # this will have softmax applied
        
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
    def __init__(self,input_size,batch_size,epochs,lr,batches_per_epoch,num_classes,numConvBlocks,kernel_size,feature_maps,embedding_dim):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.batches_per_epoch = batches_per_epoch
        self.num_classes = num_classes

        self.odeBlock = ODEBlock(ODEFunc(input_size=input_size,numConvBlocks=numConvBlocks,kernel_size=kernel_size,feature_maps=feature_maps,embedding_dim=embedding_dim,num_classes=num_classes))
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

    convNode = convNODENET(features.shape[1],batch_size=1,epochs=10,lr=0.001,batches_per_epoch=1,num_classes=10,numConvBlocks=1,kernel_size=3,feature_maps=[64,128,256,512],embedding_dim=100)
    convNode.fit(features,response)