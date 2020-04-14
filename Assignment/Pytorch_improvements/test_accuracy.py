# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Print test accuracy:
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
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
num_classes = 10
batch_size = 200
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

# Hyperparameters combination
num_layers = 5
hidden_size = n_neurons(num_layers)
activation_func = nn.LeakyReLU()
batch_norm_layer = True
drop_layer = True

# Model
hidden_size = n_neurons(num_layers)
model = MultiLayerPerceptronGridSearch(input_size, hidden_size, num_classes, activation_func=activation_func,
                                        batchn=batch_norm_layer, dropout=drop_layer).to(device)

# Run the test code once you have your by setting train flag to false
# and loading the best model

best_model = None
best_model = torch.load('best_model.ckpt', map_location=torch.device('cpu'))

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
        images = images.view(images.size(0), -1) # reshape input
        predicted = torch.argmax(model(images), dim=1) # find class


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('\nAccuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
