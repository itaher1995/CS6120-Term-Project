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

import sys
sys.path.append("..")
import config   # Read in hyperparameters from config so that we can control for input size in networks later
import data_util

# Get batch data

# Create ODE network based on pytorch objects in ODEfunc class
	# Needs to have at least __init__ and forward functions
	# Where __init__ has the network declaration and forward has the forward propogation

# Sets to GPU if possible
# Might be wrong device name for gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

is_print = False

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

# Encodes y values into onehot vector
def onehot_encoder(y, num_classes):
	# Intialize classes
	onehot_vector = np.zeros(num_classes)

	# Assign appropriate class as 1
	onehot_vector[y] = 1

	return onehot_vector


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        print(f'dims: {dim_in} {dim_out}')
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        if is_print:
            print('ConcatConv2d foward xshape', x.shape)
        tt = torch.ones_like(x[:, :1, :, :]) * t
        if is_print:
            print('ConcatConv2d foward 1', tt.shape)
        ttx = torch.cat([tt, x], 1)
        if is_print:
            print('ConcatConv2d foward 2', ttx.shape)
        l = self._layer(ttx)
        if is_print:
            print('Layer ttx')
        return l


class ODEFunc(nn.Module):
    
    def __init__(self,emb_size,kern1,kern2,kern3,num_filters,num_classes):
        super().__init__()
        
        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=[kern1,emb_size])
        # self.conv2 = nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=[kern2,emb_size])
        # self.conv3 = nn.Conv2d(in_channels=1,out_channels=num_filters,kernel_size=[kern3,emb_size])
        if is_print:
            print(f'input: {num_filters}, {kern1}')

        self.conv1 = ConcatConv2d(1, num_filters, ksize=[kern1, config.DIM_EMBEDDING])
        self.conv2 = ConcatConv2d(1, num_filters,ksize=[kern2, config.DIM_EMBEDDING])
        self.conv3 = ConcatConv2d(1, num_filters,ksize=[kern3, config.DIM_EMBEDDING])


          
        

    def forward(self, t, x):

        # x1 = torch.squeeze(F.relu(self.conv1(x)),-1)
        # x2 = torch.squeeze(F.relu(self.conv2(x)),-1)
        # x3 = torch.squeeze(F.relu(self.conv3(x)),-1)
        # print(x1.shape)

        
        # x1 = F.max_pool1d(x1,x1.size(2))
        # x2 = F.max_pool1d(x2,x2.size(2))
        # x3 = F.max_pool1d(x3,x3.size(2))

        
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)



        # out = torch.cat([x1,x2,x3],2) # shape is (input_size*3,num_filters)
        # print(out.shape)
        # out = out.view(out.size(0),-1)
        # print('1', out.shape)
        if is_print:
            print('conv1')
        x1 = self.conv1(t, x)
        if is_print:
            print('conv2')
        x2 = self.conv2(t, x)
        if is_print:
            print('conv3')
        x3 = self.conv3(t, x)

        out = torch.cat([x1,x2,x3],2) # shape is (input_size*3,num_filters)
        #out = out.view(out.size(0),-1)

        if is_print:
            print('Did out work?', x1.shape)
            print(out.shape)
        
        return x1

class ODEBlock(nn.Module): # adapted from rtqichen
    def __init__(self, odefunc):
        
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

        
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        
        out = odeint_adjoint(self.odefunc, x, self.integration_time)
        if is_print:
            print('pls halp', len(out[1]))
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
    def __init__(self,emb_size,batch_size,lr,epochs,kern1,kern2,kern3,num_filters,num_classes):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.batches_per_epoch = 1
        self.num_classes = num_classes

        # Features sizes and dimensions need to be decreased before going into ode layers
        downsampling_layers = [
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=emb_size)
        ]

        # The structure for the novel ode method
        odeBlock = ODEBlock(ODEFunc(emb_size,kern1,kern2,kern3,num_filters,num_classes))
        
        # Fully connected layers for output of function
        fc_layers = [
            nn.Linear(in_features = num_filters*3, out_features = num_classes)#,
            #F.log_softmax(dim=num_classes)
        ]

        print(num_filters * 3, num_classes)

        # Initialize end to end model
        #self.model = nn.Sequential(*downsampling_layers, odeBlock, *fc_layers).to(device)
        self.model = nn.Sequential(odeBlock, *fc_layers).to(device)


    def fit(self):
        loader = data_util.load_data()
        data_iter = data_util.inf_generator(loader)

        criterion = nn.CrossEntropyLoss().to(device)
        
        lr_fn = learning_rate_with_decay(
        self.lr, self.batch_size, batch_denom=128, batches_per_epoch=self.batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    	)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        for itr in range(self.epochs * self.batches_per_epoch):
            
            X, y = data_iter.__next__()
            X=[x.numpy()[0] for x in X] 

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(itr)

            #X_batch = X.sample(n=self.batch_size)
            #y_batch = [onehot_encoder(v,10) for v in y[X_batch.index].values]

            #X = torch.FloatTensor([X_batch.values]).to(device)#.cuda() # have to convert to tensor
            #Y = torch.FloatTensor(y_batch).long().to(device)#.cuda()
            X = Variable(torch.FloatTensor([X]), requires_grad=True).to(device) # have to convert to tensor
    
            y = Variable(torch.LongTensor([y]), requires_grad=False).to(device)
            #X = X#.to(device)
            #y = y#.to(device)

            X = X.unsqueeze(1)

            optimizer.zero_grad()
            
            logits = self.model(X)
            print(len(logits))
            print(logits[0])
            print(y)
            
            loss = criterion(logits[0], y)
    
            self.odeBlock[0].nfe = 0
    
            loss.backward()
            optimizer.step()
            
            self.odeBlock[0].nfe = 0
            
    def predict(self,X):
        #predX = Variable(torch.FloatTensor(X.values)).to(device)
        predX = torch.FloatTensor(X.values).to(device)
        prob = self.model(predX)
        return prob
        

if __name__=='__main__':
    convNode = convNODENET( config.DIM_EMBEDDING,config.BATCH_SIZE,0.001,80,1,2,3,1,20)
    convNode.fit()