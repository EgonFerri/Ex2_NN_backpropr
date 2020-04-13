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
    Activation = 'Tanh()'
elif Activation == 'relu':
    Activation = 'ReLU()'
else:
    Activation = 'LeakyReLU()'

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
train_loss = np.load('metrics/'+idx+'_train_loss.npy')
train_acc = np.load('metrics/'+idx+'_train_acc.npy')
val_loss = np.load('metrics/'+idx+'_val_loss.npy')
val_acc = np.load('metrics/'+idx+'_val_acc.npy')
lr = np.load('metrics/'+idx+'_learning_rate.npy')

# Plot validation loss
plt.figure(figsize=(10,3))
x = np.arange(len(val_loss))
lim = len(min([train_loss,val_loss], key=len))
train_loss = train_loss[:lim]
val_loss = val_loss[:lim]
xi = list(range(len(x)))
plt.plot(x, train_loss, label='Train loss')
plt.plot(x, val_loss, label='Validation loss')
plt.xticks(xi, x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', size = 15)
plt.ylabel('Loss', size = 15, labelpad=20)
plt.legend()
plt.show()

# Plot validation accuracy
plt.figure(figsize=(10,3))
x = np.arange(len(val_acc))
lim = len(min([train_acc,val_acc], key=len))
train_acc = train_acc[:lim]
val_acc = val_acc[:lim]
xi = list(range(len(x)))
plt.plot(x, train_acc, label='Train accuracy')
plt.plot(x, val_acc, label='Validation accuracy')
plt.xticks(xi, x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', size = 15)
plt.ylabel('Classification accuracy', size = 15, labelpad=20)
plt.legend()
plt.show()

# # Plot learning rate
plt.figure(figsize=(10,3))
x = np.arange(len(lr))
xi = list(range(len(x)))
plt.plot(x, lr, label='Learning rate')
plt.xticks(xi, x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Epoch', size = 15)
plt.ylabel('Learning rate', size = 15, labelpad=20)
plt.legend()
plt.show()
