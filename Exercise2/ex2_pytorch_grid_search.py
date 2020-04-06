import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm_notebook as tqdm

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

#-------------------------------------------------
# Class for the model of grid search
#-------------------------------------------------

class MultiLayerPerceptronGridSearch(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, num_layers, activation_func, dropout=False):
        super(MultiLayerPerceptronGridSearch, self).__init__() # it writes initialization for nn.Module class as well
        #################################################################################
        # TODO: Initialize the modules required to implement the mlp with the layer     #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        # Make use of linear and relu layers from the torch.nn module                   #
        #################################################################################

        layers = [] #Use the layers list to store a variable number of layers

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # input
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(activation_func)
        if dropout == True: layers.append(nn.Dropout(p=0.3))

        # following layers
        for layer in range(num_layers-1):
          layers.append(nn.Linear(hidden_layers[0], hidden_layers[0]))
          layers.append(activation_func)
          if dropout == True: layers.append(nn.Dropout(p=0.3))

        # output
        layers.append(nn.Linear(hidden_layers[0], num_classes))


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Enter the layers into nn.Sequential, so the model may "see" them
        # Note the use of * in front of layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        # Note that you do not need to use the softmax operation at the end.            #
        # Softmax is only required for the loss computation and the criterion used below#
        # nn.CrossEntropyLoss() already integrates the softmax and the log loss together#
        #################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.layers(x)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return out

#-------------------------------------------------
# Run grid search
#-------------------------------------------------
torch.manual_seed(123)
hyper_grid = {'num_layers': [2, 3, 4, 5, 10, 15], 'activation_func' : [nn.ReLU(), nn.PReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()], 'flag':[True, False]}
grid = ParameterGrid(hyper_grid)
all_res = []
for hypers in tqdm(grid):

  # hyperparameters combination
  num_layers = hypers['num_layers']
  activation_func = hypers['activation_func']
  flag = hypers['flag']
  print('\n\n\n\nHyperparameters: num_layers = '+str(num_layers)+', activation_func = '+str(activation_func)+', flag = '+str(flag))

  # initialize model
  model = MultiLayerPerceptronGridSearch(input_size, hidden_size, num_classes, num_layers, activation_func, dropout=flag).to(device)

  model.apply(weights_init)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

  # Train the model
  lr = learning_rate
  total_step = len(train_loader)
  for epoch in range(num_epochs):
      correct_train = 0
      total_train = 0
      model.train()
      for i, (images, labels) in enumerate(train_loader):
          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)
          #################################################################################
          # TODO: Implement the training code                                             #
          # 1. Pass the images to the model                                               #
          # 2. Compute the loss using the output and the labels.                          #
          # 3. Compute gradients and update the model using the optimizer                 #
          # Use examples in https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
          #################################################################################
          # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          # Pass images to the model to compute predicted labels
          images = images.view(-1, input_size)
          pred_labels = model(images)


          # Compute the loss using the predicted labels and the actual labels.
          loss = criterion(pred_labels, labels)

          # Compute gradients and update the model using the optimizer
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Compute train accuracy
          predicted = torch.argmax(pred_labels, dim=1)
          total_train += labels.size(0)
          correct_train += (predicted == labels).sum().item()

          # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

          if (i+1) % 100 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

      # Code to update the lr
      lr *= learning_rate_decay
      update_lr(optimizer, lr)
      with torch.no_grad():
          correct = 0
          total = 0
          model.eval()
          for images, labels in val_loader:
              images = images.to(device)
              labels = labels.to(device)
              ####################################################
              # TODO: Implement the evaluation code              #
              # 1. Pass the images to the model                  #
              # 2. Get the most confident predicted class        #
              ####################################################
              # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
              images = images.view(-1, input_size)
              predicted = torch.argmax(model(images), dim=1)


              # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

          print('Validataion accuracy is: {} %'.format(100 * correct / total))
  all_res.append({'N. layers': num_layers, 'Activation': str(activation_func), 'Dropout': flag, 'train_acc': (100 * correct_train / total), 'val_acc': (100 * correct / total)})

# Save result in a dataframe
out_rs = pd.DataFrame(all_res)
out_rs.to_csv('MLP_grid_search.csv', index=False)
print('Grid search results saved')
