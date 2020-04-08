from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



class TwoLayerNetAdvanced(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    Is trained with adam gradient-based optimization algorithm.
    Is possible to introduce a dropout
    """



    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary. We also have to store other
        variables needed for adam.
        """
        
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.sqrs = {}
        self.sqrs['W1'] = 0
        self.sqrs['b1'] = 0
        self.sqrs['W2'] = 0
        self.sqrs['b2'] = 0

        self.v = {}
        self.v['W1'] = 0
        self.v['b1'] = 0
        self.v['W2'] = 0
        self.v['b2'] = 0

    def loss(self, X,p,  y=None, reg=0.0, dropout=False):
        """
        Compute the loss and gradients for a two-layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2'] #shapes 10,3 -- 3
        N, D = X.shape

        # Compute the forward pass
        scores = 0.

        def ReLU(z):
          return np.maximum(0, z)

        def softmax(z):
          e=np.exp(z)
          return  e/e.sum(axis=1, keepdims=True)
        
        a1 = X
        z2 = a1.dot(W1) + b1
        a2 = ReLU(z2)
        if dropout==True:
          u1 = (np.random.rand(*a2.shape)<p)/p
          a2 *= u1
        z3 = a2.dot(W2) + b2
        scores = softmax(z3)
        

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores


        # Compute the loss
        loss = 0.
        
        
        # Implement the loss for the softmax output layer
        
        
        like = scores[range(N), y] #probabilities for the true labels
        log_like = - np.log(like)
        loss_no_reg = np.sum(log_like)/N

        regula = reg* (np.linalg.norm(W1)**2+np.linalg.norm(W2)**2)

        loss = loss_no_reg + regula       

        # Backward pass: compute gradients
        grads = {}

        kron = np.zeros((scores.shape[0],scores.shape[1]))
        kron[range(N), y] = 1
        grads["W2"] = (1/N) * (scores - kron).T.dot(a2).T + 2*reg*W2
        b2_der = np.ones(N)
        grads["b2"] = (1/N) * (scores - kron).T.dot(b2_der)
        grads["W1"] = (1/N) * ((scores - kron).dot(W2.T)* np.where(z2 < 0,0,1)).T.dot(a1).T + 2*reg*W1
        b1_der = np.ones(z2.shape[0])
        grads["b1"] = (1/N) * ((scores - kron).dot(W2.T)* np.where(z2 < 0,0,1)).T.dot(b1_der)

        return loss, grads



    def train(self, X, y,p, X_val, y_val,beta1, beta2, eps,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, dropout=False, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        
        num_train = X.shape[0]
        iterations_per_epoch = max( int(num_train // batch_size), 1)


        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = X
            y_batch = y
            
            index=np.random.randint(low=0, high=num_train, size=batch_size)
            X_batch=X[index]
            y_batch=y[index]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch,p=p, y=y_batch, reg=reg, dropout=dropout)
            loss_history.append(loss)

            
            # update parameters using adam
            for param in self.params.keys():
              g = grads[param]/batch_size
              self.v[param] = beta1 * self.v[param] + (1. - beta1) * g
              self.sqrs[param] = beta2 * self.sqrs[param] + (1. - beta2) * np.square(g)
              sqr2 = self.sqrs[param] / (1 - np.power(beta2 ,it+1))
              v2 = self.v[param] / (1 - np.power(beta1 ,it+1))
              self.params[param] -= learning_rate * v2 / (np.sqrt(sqr2) + eps)

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # At every epoch check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }



    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        def ReLU(z):
          return np.maximum(0, z)

        def softmax(z):
          e=np.exp(z)
          return  e/e.sum(axis=1, keepdims=True)

        a1 = X
        z2 = a1.dot(W1) + b1
        a2 = ReLU(z2)
        z3 = a2.dot(W2) + b2
        scores = softmax(z3)

        y_pred=np.argmax(scores, axis=1)


        return y_pred


