# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Take hyperparameters as inputs

num_layers = int(input('Number of layers, choose among [2,3,4,5,10]: '))
assert num_layers in [2,3,4,5,10]

Activation = input('Choose an activation fucntion among [tanh, relu, leakyrelu]: ')
assert Activation in ['tanh', 'relu', 'leakyrelu']
if Activation == 'tanh':
    Activation = nn.Tanh()
elif Activation == 'relu':
    Activation = nn.ReLU()
else:
    Activation = nn.LeakyReLU()

Optimizer = input('Choose an optimizer among [SGD, Adam]: ')
assert Optimizer in ['SGD', 'Adam']

Batchn = input('Insert Batch Nomalization? [y/n]: ')
assert Batchn in ['y','n']
if Batchn == 'y':
    Batchn = True
else:
    Batchn = False

Dropout = input('Insert Dropout? [y/n]: ')
assert Dropout in ['y','n']
if Dropout == 'y':
    Dropout = True
else:
    Dropout = False

# Hyperparameter combination
print('\nHyperparameter combination')
print('Number of layers:', num_layers)
print('Activation:', Activation)
print('Optimizer:', Optimizer)
print('Batchn:', Batchn)
print('Dropout:', Dropout,'\n')

# Find index corresponding to the chosen combination
df = pd.read_csv('MLP_grid_search.csv')
idx = str(df.index[(df['N. layers'] == num_layers) & (df['Activation'] == str(Activation)) & (df['Optimizer'] == Optimizer)
                & (df['Dropout'] == Dropout) & (df['Batch Normalization'] == Batchn)].tolist()[0])

# Load result for plots
val_acc = np.load('metrics/'+idx+'_val_acc.npy')
val_loss = np.load('metrics/'+idx+'_val_loss.npy')
lr = np.load('metrics/'+idx+'_learning_rate.npy')

# Plot validation loss
plt.figure(figsize=(10,5))
x = np.arange(len(val_loss))
xi = list(range(len(x)))
plt.plot(x, val_loss)
plt.xticks(xi, x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', size = 15)
plt.ylabel('Validation loss', size = 15, labelpad=20)
plt.show()

# Plot validation accuracy
plt.figure(figsize=(10,5))
x = np.arange(len(val_acc))
xi = list(range(len(x)))
plt.plot(x, val_acc)
plt.xticks(xi, x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', size = 15)
plt.ylabel('Validation accuracy', size = 15, labelpad=20)
plt.show()

# # Plot learning rate
plt.figure(figsize=(10,5))
x = np.arange(len(lr))
xi = list(range(len(x)))
plt.plot(x, lr)
plt.xticks(xi, x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', size = 15)
plt.ylabel('Learning rate', size = 15, labelpad=20)
plt.show()

"""
# Print test accuracy:
def kaiming_init(m):
  '''Initilize weights (with kaiming initialization)
  and bias (with 0s)'''
  if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.)

def xavier_init(m):
  '''Initilize weights (with kaiming initialization)
  and bias (with 0s)'''
  if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    '''Update learning rate'''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def n_neurons(num_layers):
  '''Given the number of layer it returns a list with legth equal to the
  number of layer. Each element of the list is an integer that represents the
  number of neuron for the layer corresponding to its index'''
  if num_layers < 2:# error
    return 'Error: number of layer has to be >= 2'
  else:
    neurons = [50]*2
    for i in range(2, num_layers):
      neurons += [50*i]
    if max(neurons) > input_size: # error
      return 'Error: number of neuron is higher than input size'
    else:
      return sorted(neurons, reverse=True)

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters (fixed)
#--------------------------------
input_size = 32 * 32 * 3
num_classes = 10
num_epochs = 20 # more epochs but stopping rule has been introduced
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000

#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
norm_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=norm_transform
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#======================================================================================
# Fully connected neural network for Grid Search
#======================================================================================
class MultiLayerPerceptronGridSearch(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, activation_func, batchn=False, dropout=False):
        super(MultiLayerPerceptronGridSearch, self).__init__()

        layers = [] # layers list to store a variable number of layers

        layers.append(nn.Linear(input_size, hidden_layers[0])) # Input layer
        if batchn:
          layers.append(nn.BatchNorm1d(num_features=hidden_layers[0])) # Batch Normalization
        layers.append(activation_func) # Activation function
        if dropout:
          layers.append(nn.Dropout(p=0.3)) # Dropout

        # iterate through hidden layer list to add as many layer as wanted
        # hidden layer is a list in which every item represent the numeber of neurons for that layer
        for i in range(1, len(hidden_layers)-1):
          layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
          if batchn:
            layers.append(nn.BatchNorm1d(num_features=hidden_layers[i])) # Batch Normalization
          layers.append(activation_func) # Activation function
          if dropout:
            layers.append(nn.Dropout(p=0.3)) # Dropout

        layers.append(nn.Linear(hidden_layers[-1], num_classes)) # Output layer

        # Enter the layers into nn.Sequential, so the model may "see" them
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        out = self.layers(x)

        return out

# Model
hidden_size = n_neurons(num_layers)
model = MultiLayerPerceptronGridSearch(input_size, hidden_size, num_classes, activation_func=Activation, batchn=Batchn, dropout=Dropout).to(device)

# Run the test code once you have your by setting train flag to false
# and loading the best model

best_model = None
best_model = torch.load('weights/'+idx+'_model.ckpt', map_location=torch.device('cpu'))

model.load_state_dict(best_model)

# Test the model
model.eval() #set dropout and batch normalization layers to evaluation mode

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        ####################################################
        # TODO: Implement the evaluation code              #
        # 1. Pass the images to the model                  #
        # 2. Get the most confident predicted class        #
        ####################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        images = images.view(images.size(0), -1) # reshape input
        predicted = torch.argmax(model(images), dim=1) # find class


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('\nAccuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
"""
