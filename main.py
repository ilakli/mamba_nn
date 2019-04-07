from mamba_net import *
import numpy as np
from random import randint
import random
import pickle
import os.path
import time

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

def create_simple_dataset():
    random.seed(0)
 
    data_x = None
    data_y = []
    for x in range(100):
        for y in range(100):

            features = np.array([y/100, x/100])
            if data_x is not None:
                data_x = np.vstack((data_x, features))
            else:
                data_x = features.copy()

            if x > y:
                data_y.append(0)
            else:
                data_y.append(1)
 
    data_x = np.transpose(data_x)

    return data_x, data_y

def get_cifar_dataset(batch_name):
    file_path = os.path.join('cifar-10', batch_name)

    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    cifar_x = np.transpose(dict[b'data'])
    cifar_y = np.array(dict[b'labels'])

    cifar_x_mean0 = cifar_x - np.mean(cifar_x, axis=0)
    cifar_x_standardized = cifar_x_mean0 / np.std(cifar_x, axis=0)

    return cifar_x_standardized, cifar_y

def main():

    kobi = MambaNet(24)

    cifar_x, cifar_y = get_cifar_dataset("data_batch_1")

    layer1 = BaseLayer(128, "relu", "xavier",
                       (quadratic_weight_function, d_quadratic_weight_function),
                       2,
                       True,
                       0.0005)
    layer2 = BaseLayer(64, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)
    layer3 = BaseLayer(10, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)
    kobi.add(layer1)
    kobi.add(layer2)
    kobi.add(layer3)

    kobi.compile(3072, 10)
    kobi.train(cifar_x, cifar_y, n_epochs = 5, learning_rate=0.01)   

if __name__ == '__main__':
    main()