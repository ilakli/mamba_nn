from mamba_net import *
import numpy as np
from random import randint
import random

def linear_weight_function(W, x):
    A = np.matmul(W[0], x)
    return A

def d_linear_weight_function(W, x, dz):
    dW = np.matmul(dz, x.T)
    dX = np.matmul(W[0].T, dz)
    return (np.array(dW), dX)

def quadratic_weight_function(W,x):
    A = np.matmul(W[0], x**2) + np.matmul(W[1], x)
    # print(A)
    return A

def d_quadratic_weight_function(W, x, dz):
    dW0 = np.matmul(dz, np.transpose(x**2))
    dW1 = np.matmul(dz, x.T)
    dX = np.matmul(W[0].T, dz) + np.matmul(W[0].T, dz)
    return (np.array([dW0, dW1]), dX)

def main():

    random.seed(0)
 
    all_features = None
    data_y = []
    for x in range(100):
        for y in range(100):

            features = np.array([y/100, x/100])
            if all_features is not None:
                all_features = np.vstack((all_features, features))
            else:
                all_features = features.copy()

            if x > y:
                data_y.append(0)
            else:
                data_y.append(1)
 
    all_features = np.transpose(all_features)
 
    kobi = MambaNet(24)
    # layer1 = BaseLayer(3, "sigmoid", "tupoi",
    #                    (linear_weight_function, d_linear_weight_function),
    #                    1,
    #                    True)
    # layer2 = BaseLayer(3, "sigmoid", "tupoi",
    #                    (linear_weight_function, d_linear_weight_function),
    #                    1,
    #                    True)
    # layer3 = BaseLayer(2, "sigmoid", "tupoi",
    #                    (linear_weight_function, d_linear_weight_function),
    #                    1,
    #                    True)
    # layer1 = BaseLayer(3, "relu", "xavier",
    #                    (linear_weight_function, d_linear_weight_function),
    #                    1,
    #                    True)
    # layer2 = BaseLayer(3, "relu", "xavier",
    #                    (linear_weight_function, d_linear_weight_function),
    #                    1,
    #                    True)
    # layer3 = BaseLayer(2, "relu", "xavier",
    #                    (linear_weight_function, d_linear_weight_function),
    #                    1,
    #                    True)
    layer1 = BaseLayer(3, "sigmoid", "tupoi",
                       (quadratic_weight_function, d_quadratic_weight_function),
                       2,
                       True)
    layer2 = BaseLayer(3, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)
    layer3 = BaseLayer(2, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True)


    kobi.add(layer1)
    kobi.add(layer2)
    kobi.add(layer3)

    kobi.compile(2, 2)

    kobi.train(all_features, data_y, n_epochs = 5000, learning_rate=0.005)

if __name__ == '__main__':
    main()