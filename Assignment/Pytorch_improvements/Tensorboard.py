import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchbearer import Trial
from torchbearer.callbacks import TensorBoard
import os


# The following code is the one used to create folder logs needed to create visualization in TensorBoard
# the code is here commented because the GPU is a requirement, run it on Google colab. However since logs folder
# has been saved it is possible to access to TensorBoard with the last line of this code.
'''
#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
hidden_size = [50]
num_classes = 10
num_epochs = 10
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
train = True

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

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron, self).__init__() # it writes initialization for nn.Module class as well

        layers = [] #Use the layers list to store a variable number of layers

        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(hidden_layers)-1):
          layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
          layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], num_classes))

        # Enter the layers into nn.Sequential, so the model may "see" them
        # Note the use of * in front of layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out

#-------------------------------------------------
# Tensorboard implementation
#-------------------------------------------------

model = MultiLayerPerceptron(input_size, hidden_size, num_classes).to(device)
lr = learning_rate
optimizer = optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=reg)
loss = nn.CrossEntropyLoss()

torchbearer_trial = Trial(model, optimizer, loss, metrics=['acc', 'loss'],
                          callbacks=[TensorBoard(write_graph=False, write_batch_metrics=False, write_epoch_metrics=True)]).to('cuda')
torchbearer_trial.with_generators(train_generator=train_loader, val_generator=val_loader)
torchbearer_trial.run(epochs=10)
'''
print('Open following link:')
os.system('tensorboard --logdir=logs')
