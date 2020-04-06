import numpy as np
import matplotlib.pyplot as plt
from two_layernet import TwoLayerNet
from gradient_check import eval_numerical_gradient
from data_utils import get_CIFAR10_data
from vis_utils import visualize_grid
from two_layernet_advanced import TwoLayerNetAdvanced
from sklearn.decomposition import PCA

num_classes = 10

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
pca = PCA(n_components = 400) 
X_train = pca.fit_transform(X_train) 
X_val = pca.transform(X_val)
X_test = pca.transform(X_test) 
best_comb = [250, 5000, 512, 0.001, 0.001]
np.random.seed(123)
p = 0.8
beta1 = 0.9
beta2 = 0.999
eps = 1e-7
best_net = TwoLayerNetAdvanced(X_train.shape[1], best_comb[0], num_classes)
stats = best_net.train(X_train, y_train,p, X_val, y_val, beta1, beta2, eps,
            num_iters=best_comb[1], batch_size=best_comb[2],
            learning_rate=best_comb[3], learning_rate_decay=0.95,
           reg=best_comb[4], verbose=True)

# Predict on the validation set
print('-----------')
print('hidden size: ',best_comb[0], ', num_iters: ',best_comb[1], ', batch size: ',best_comb[2], ', learning_rate: ', best_comb[3], ', regula: ',best_comb[4] )
val_acc = (best_net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)
print('------------')

plt.figure(7)
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
