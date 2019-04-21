# CS6120-Term-Project
This is the github repository for **Text Classification Using Neural Ordinary Differential Equations**. The project was completed by Forrest Hooton and Ibrahim Taher, both MS in Data Science students at Northeastern University. The project was completed for their natural language processing class. 

## Dependencies

In order for this code to run you will need the following:

pandas >= 0.24.2
numpy >= 1.13.3
pickle
time
torch >= 1.0.1
sys
os
warnings



## Getting the data

The dataset used for this project was the 20newsgroup dataset. 

In order to get the files needed for this project first create a folder **data**. This folder will serve as the location from which the models will retrieve, training, validation and testing data. Getting the dataset in the form needed for the project can be done by running the file **data_pipe.py** located in the **processing** folder. This will then create the file **newsgroup_CV.pkl** which will be located in the data folder of your local repostory.

## Running the models

The scripts for the models are located in the **models** directory. In that directory there are several python files, but the two of relevance are named **basic_model.py** and **node.py**. The first file serves as the code for our baseline network, a feed forward neural network. The second is the experimental network, in which we use neural ODE's to solve for the weights. If you run the code, it will do hyperparameter searching for the baseline model and then use the best results (where the hidden layers neurons = 200) for the NODE model. We also save the weights of the models in a folder called **model_train_weights**.

## API Reference

### Models

### FFN

**FFN**

Feedforward neural network implementation in PyTorch. 

INPUTS: hidden_size - (int) number of neurons for each hidden layer
        
numClasses - (int) number of classes for the model. In this it is 20.

lr - (float) the learning rate for the model

epochs - (int) How many iterations will the model learn for.

**fit**

Function to train the neural network.

INPUTS: name - (str) filename for saving the file. Saves to csv. No file type needed.

save_weights - Saves weights for this model post training.

**score**

Will output the top 1 accuracy score for this model

**test**

Will output model's top-1 predictions on a predefined test set.

INPUTS: name - (str) filename for saving the file. Saves to csv. No file type needed.

### NODE

**FFN**

Creates the FFN architecture using Neural Ordinary Differential Equation Solvers.

INPUTS: hidden_size - (int) number of neurons for each hidden layer
        
numClasses - (int) number of classes for the model. In this it is 20.

lr - (float) the learning rate for the model

epochs - (int) How many iterations will the model learn for.

**fit**

Function to train the neural network.

INPUTS: name - (str) filename for saving the file. Saves to csv. No file type needed.

save_weights - Saves weights for this model post training.

**score**

Will output the top 1 accuracy score for this model

**test**

Will output model's top-1 predictions on a predefined test set.

INPUTS: name - (str) filename for saving the file. Saves to csv. No file type needed.







