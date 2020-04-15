# Ex2_NN_backpropr

The **Report.pdf** file, as can be seen, contains the report. The **assignement** folder includes all the codes and file developed to answer the 4 assignment questions. More specifically, the folder is organized as follow:  

### for Question 1 
- **two_layernet.py** has the code needed to run the two layer network for the forward pass.

### for Question 2 
- **two_layernet.py** contains also the backpropagation algorithm.
- **Ex2_FCnet.py** includes the check on the gradients and on the loss.
- **vis_utils.py** has some useful functions for visualization.
- **gradient_check.py** includes the functions to evaluate the numerical gradients.
- **get_datasets** loads the dataset

### for Question 3  
- **Ex2_FCnet.py** also includes the grid search for the hyperparameters, and the best net evaluation.
- **two_layernet_advanced.py** is a file with the advanced two layer network (PCA-Adam-Dropout).
- **test.py** is used to test the modified two layer network.
- **datasets** is the cifar-10 dataset divided in batches.

### for Question 4  
- **ex2_pytorch.py**, contains the code to implement the two-layer network with Pytorch.  
- **model.ckpt** are the optimized weights produced by running ex2_pytorch.py.  
- **Pytorch_improvements**, this folder includes its own README.md in which is specified the organization of the folder, basically it contains improvements to the Pytorch NN implemented in ex2_pytorch.py and some visualization tools.  
