
List of file and folders related to Grid Search:  

- *ex2_pytorch_grid_search.py* is the python file with the code for the grid search (just run it to execute the grid search).  

- *Check_combinations.py*, this file is useful to show result obtained with grid search. Just run it and input the hyperparameter required (120 possible combinations), quickly validation loss/accuracy and learning rate plots will be printed. This was possible because this file retrieves information needed from other files present in this folder.  

- *MLP_grid_search.csv* is a csv file with results of grid search, i.e. combination of hyperparameters with the corresponding validation accuracy value.  

- *metrics*, this folder contains *.npy* files (use `numpy load` to load them back as *type list*). Each file representing learning rate or train/validation loss/accuracy for each epoch is associated (through and index present in the file name) to a combination of hyperparameters.  

- *data*, this folders contains *CIFAR10* image dataset.  


List of file and folders related to Grid Search:

- *Tensorboard.py*, in this file is implemented the basic version of two-layered MPL using *tensorboardX* and *torchbearer* libraries. In this way it was possible to create the *logs* folder used to have some visualization in *TensorBoard*. To open *TensorBoard* panel, just run this file and open the resulting link.   

- *logs*, folder with files necessary to create visualizations in *TensorBoard*.
