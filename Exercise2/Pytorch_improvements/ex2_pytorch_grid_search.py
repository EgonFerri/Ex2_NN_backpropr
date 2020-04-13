import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm_notebook as tqdm

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

#-------------------------------------------------
# Implementation of Grid Search
#-------------------------------------------------

# Seed for reproducibility
torch.manual_seed(2020)

# Possible values of other hyperparameters
hyper_grid = {'num_layers': [2, 3, 4, 5, 10],
              'init_&_activation_func' : [(xavier_init, nn.Tanh()), (kaiming_init, nn.ReLU()), (kaiming_init, nn.LeakyReLU())],
              'opt': [torch.optim.SGD, torch.optim.Adam],
              'batch_norm_layer': [True, False],
              'drop_layer': [True, False]}

# Form all possible combiantions
grid = ParameterGrid(hyper_grid)

all_res = []
counter = -1
for hypers in tqdm(grid):

  counter += 1

  # Hyperparameters combination
  num_layers = hypers['num_layers']
  # len(hidden_size) == num_layers and each item is the n. of neurons for that layer
  hidden_size = n_neurons(num_layers)
  init, activation_func = hypers['init_&_activation_func'] # initialiation and activation fuction to be used
  opt = hypers['opt'] # optimizer
  batch_norm_layer = hypers['batch_norm_layer'] # boolian to insert Batch Nomalization
  drop_layer = hypers['drop_layer'] # boolian to insert Dropout
  print('\n\n\n\nHyperparameters: num_layers = '+str(num_layers)+', activation_func = '+str(activation_func)+
        ', batch_norm_layer = '+str(batch_norm_layer)+', drop_layer = '+str(drop_layer))

  # Model
  model = MultiLayerPerceptronGridSearch(input_size, hidden_size, num_classes, activation_func=activation_func,
                                         batchn=batch_norm_layer, dropout=drop_layer).to(device)

  # Weight initialization
  model.apply(init)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = opt(model.parameters(), lr=learning_rate, weight_decay=reg)

  # Train the model
  lr = learning_rate
  total_step = len(train_loader)

  # To early stop
  best_score = None
  patience = 5

  # To track training accuracy/loss
  train_acc = []
  train_loss = []
  # To track the validation accuracy/loss
  val_acc = []
  val_loss = []
  # To track learning rate
  lr_list = []

  for epoch in range(num_epochs):
      model.train() #set dropout and batch normalization layers to training mode
      correct_tr = 0
      total_tr = 0
      for i, (images, labels) in enumerate(train_loader):
          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)

          # Pass images to the model to compute predicted labels
          images = images.view(images.size(0), -1)
          pred_labels = model(images)

          # Compute the loss using the predicted labels and the actual labels.
          loss = criterion(pred_labels, labels)

          # Compute gradients and update the model using the optimizer
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          tr_loss = loss.item()
          if (i+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, tr_loss))

      # Train accuracy
      predicted = torch.argmax(pred_labels, dim=1)
      total_tr += labels.size(0)
      correct_tr += (predicted == labels).sum().item()
      acc_tr = correct_tr / total_tr
      train_acc.append(acc_tr)
      train_loss.append(round(tr_loss, 4))

      # Code to update the lr
      lr *= learning_rate_decay
      lr_list.append(lr)
      update_lr(optimizer, lr)
      with torch.no_grad():
          correct = 0
          total = 0
          model.eval() #set dropout and batch normalization layers to evaluation model
          for images, labels in val_loader:
              images = images.to(device)
              labels = labels.to(device)

              images = images = images.view(images.size(0), -1)
              pred_labels = model(images)
              v_loss = criterion(pred_labels, labels) # validation loss
              predicted = torch.argmax(pred_labels, dim=1)

              total += labels.size(0)
              correct += (predicted == labels).sum().item()
          acc = correct / total
          print('Validataion accuracy is: {} %'.format(100 * acc))

          # Stopping rule
          if best_score is None: # occurs at fist epoch
            best_score = acc # set a value for best score
            count = 0 # count needed to check patience
            torch.save(model.state_dict(), str(counter)+'_model.ckpt') # save weights
            print('Model saved')
          elif best_score > acc + 1e-03: # best accuracy is better than the current accuracy
            count += 1 # update count
            if count >= patience: # if count reach patience -> early stop
              print('Patience: '+str(count)+'/'+str(patience)+' -> Early stopping')
              break
            else:
              print('Patience: '+str(count)+'/'+str(patience))
          else:
            best_score = acc # update best score
            count = 0 # set/reset count to 0
            torch.save(model.state_dict(), str(counter)+'_model.ckpt')  # save weights
            print('Model saved')

      # Save accuracy value for the current epoch
      val_acc.append(acc)
      val_loss.append(round(v_loss.item(), 4))

  # Save loss and accuracy
  np.save(str(counter)+'_learning_rate.npy', lr_list)
  np.save(str(counter)+'_train_acc.npy', train_acc)
  np.save(str(counter)+'_train_loss.npy', train_loss)
  np.save(str(counter)+'_val_acc.npy', val_acc)
  np.save(str(counter)+'_val_loss.npy', val_loss)

  # Save result in a dataframe
  all_res.append({'N. layers': num_layers, 'Activation': str(activation_func), 'Optimizer': str(opt), 'Dropout': drop_layer, 'Batch Normalization': batch_norm_layer,
                  'val_acc': (100 * best_score)})
  out_rs = pd.DataFrame(all_res)
  out_rs.to_csv('MLP_grid_search.csv', index=False)

print('Grid search ended and results saved')
