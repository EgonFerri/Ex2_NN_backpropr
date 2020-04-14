import numpy as np
import matplotlib.pyplot as plt
from two_layernet import TwoLayerNet
from gradient_check import eval_numerical_gradient
from data_utils import get_CIFAR10_data
from vis_utils import visualize_grid
from two_layernet_advanced import TwoLayerNetAdvanced
from sklearn.decomposition import PCA
import time


def plot(name):
    plt.figure(dpi=250, figsize=(6, 4))
    plt.suptitle(name, fontsize=14, y=1 )
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_x_overleaf/'+name+'.jpeg', dpi=500)

num_classes = 10

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
best_comb = [250, 5000, 256, 0.001, 0.001]

#dropout parameter
p = 0.8 

#adam parameters
beta1 = 0.9
beta2 = 0.999
eps = 1e-7

#################### JUST ADAM ####################################
print('------------')
print("JUST ADAM")
np.random.seed(123)

start = time.time()
best_net = TwoLayerNetAdvanced(X_train.shape[1], best_comb[0], num_classes)
stats = best_net.train(X_train, y_train,p, X_val, y_val, beta1, beta2, eps,
            num_iters=best_comb[1], batch_size=best_comb[2],
            learning_rate=best_comb[3], learning_rate_decay=0.95,
           reg=best_comb[4],dropout=False, verbose=False)
print("code execution time: ",time.time() - start)

plot('JUST ADAM')

val_acc = (best_net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
print('------------')

#################### ADAM + DROPOUT ###############################
print("ADAM+DROPOUT")
np.random.seed(123)

start = time.time()
best_net = TwoLayerNetAdvanced(X_train.shape[1], best_comb[0], num_classes)
stats = best_net.train(X_train, y_train,p, X_val, y_val, beta1, beta2, eps,
            num_iters=best_comb[1], batch_size=best_comb[2],
            learning_rate=best_comb[3], learning_rate_decay=0.95,
           reg=best_comb[4],dropout=True, verbose=False)
print("code execution time: ",time.time() - start)

plot('ADAM+DROPOUT')

val_acc = (best_net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
print('------------')

#################### ADAM + PCA  ##################################
pca = PCA(n_components = 400)
X_train = pca.fit_transform(X_train) 
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

print("ADAM+PCA")
np.random.seed(123)

start = time.time()
best_net = TwoLayerNetAdvanced(X_train.shape[1], best_comb[0], num_classes)
stats = best_net.train(X_train, y_train,p, X_val, y_val, beta1, beta2, eps,
            num_iters=best_comb[1], batch_size=best_comb[2],
            learning_rate=best_comb[3], learning_rate_decay=0.95,
           reg=best_comb[4],dropout=False, verbose=False)
print("code execution time: ",time.time() - start)

plot('ADAM+PCA')

val_acc = (best_net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
print('------------')

#################### ADAM + DROPOUT + PCA #########################
print("ADAM+PCA+DROPOUT")
np.random.seed(123)

start = time.time()
best_net = TwoLayerNetAdvanced(X_train.shape[1], best_comb[0], num_classes)
stats = best_net.train(X_train, y_train,p, X_val, y_val, beta1, beta2, eps,
            num_iters=best_comb[1], batch_size=best_comb[2],
            learning_rate=best_comb[3], learning_rate_decay=0.95,
           reg=best_comb[4],dropout=True, verbose=False)
print("code execution time: ",time.time() - start)

plot('ADAM+PCA+DROPOUT')

val_acc = (best_net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
print('------------')

#################### ADAM + DROPOUT + PCA + INCREASE BATCH ########
print("ADAM+PCA+DROPOUT+512batch")
np.random.seed(123)

start = time.time()
best_net = TwoLayerNetAdvanced(X_train.shape[1], best_comb[0], num_classes)
stats = best_net.train(X_train, y_train,p, X_val, y_val, beta1, beta2, eps,
            num_iters=best_comb[1], batch_size=best_comb[2]*2,
            learning_rate=best_comb[3], learning_rate_decay=0.95,
           reg=best_comb[4],dropout=True, verbose=False)
print("code execution time: ",time.time() - start)

plot('ADAM+PCA+DROPOUT+512batch')

val_acc = (best_net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
print('------------')





