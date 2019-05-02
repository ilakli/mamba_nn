from mamba_net import *
import numpy as np
from random import randint
import os.path
import time
from functools import reduce
from datasets import *

def linear_weight_function(W, x):
    A = np.matmul(W[0], x)
    return A

def d_linear_weight_function(W, x, dz):
    dW = np.matmul(dz, x.T)
    dX = np.matmul(W[0].T, dz)
    return (np.array(dW), dX)

def quadratic_weight_function(W, x):
    A = np.matmul(W[0], x**2) + np.matmul(W[1], x)
    return A

def d_quadratic_weight_function(W, x, dz):
    dW0 = np.matmul(dz, np.transpose(x**2))
    dW1 = np.matmul(dz, x.T)
    dX = 2 * np.matmul(W[0].T, dz) * x + np.matmul(W[1].T, dz)
    return (np.array([dW0, dW1]), dX)

def splitting_function_0(X):
    return np.zeros((X.shape[1]))

def splitting_function_mean(X):
    return (np.mean(X, axis=0) > 0.5).astype(int)

def splitting_function_even_odd(X):
    return np.arange(X.shape[1]) % 2

def main():

    kobi = MambaNet(24)
    
    val_x, val_y = get_cifar_dataset("test_batch")

    layer_21 = BaseLayer(64, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)
    layer_22 = BaseLayer(64, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)

    layer1 = BaseLayer(128, "sigmoid", "tupoi",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       False,
                       0.0005)
    layer2 = BaseLayer(64, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)
    # layer2 = BasePieceWiseLayer([layer_21, layer_22], splitting_function_even_odd)
    layer3 = BaseLayer(10, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)
    kobi.add(layer1)
    kobi.add(layer2)
    kobi.add(layer3)

    kobi.compile(3072, 10)

    file_names = ["data_batch_%s" % (str(ind)) for ind in range(1, 6)]

    kobi.train_from_files(file_names, "test_batch", get_cifar_dataset, learning_rate=0.05, n_epochs=50)

    # print (kobi.layers[1].average_output / kobi.layers[1].number_of_examples)

if __name__ == '__main__':
    main()