from mamba_net import *
import numpy as np
from random import randint
import os.path
import time
from functools import reduce, partial
from datasets import *

DEBUG_MODE = False

# Linear weight function
# Arguments:
#   x: input matrix
#   W: matrix of weight matrices (base_layer.weights)
# Returns: W*x  
def linear_weight_function(W, x):
    A = np.matmul(W[0], x)
    return A

# Derivative of linear weight function,
# gives gradient for weights and inputs.
# Arguments:
#   x:  input matrix
#   W:  matrix of weight matrices (base_layer.weights)
#   dz: derivative of this layer. 
# Returns: (dW, dx)  
def d_linear_weight_function(W, x, dz):
    dW = np.matmul(dz, x.T)
    dX = np.matmul(W[0].T, dz)
    return (np.array(dW), dX)

# Quadratic weight function
# Arguments:
#   x: input matrix
#   W: matrix of weight matrices (base_layer.weights)
# Returns: W0*(x^2) + W1*x  
def quadratic_weight_function(W, x):
    A = np.matmul(W[0], x**2) / 100 + np.matmul(W[1], x)
    return A

# Derivative of quadratic weight function, 
# gives gradient for weights and inputs.
# Arguments:
#   x:  input matrix
#   W:  matrix of weight matrices (base_layer.weights)
#   dz: derivative of this layer. 
# Returns: (dW, dx)  
def d_quadratic_weight_function(W, x, dz):
    dW0 = np.matmul(dz, np.transpose(x**2))
    dW1 = np.matmul(dz, x.T)
    dX = 2 * np.matmul(W[0].T, dz) * x / 100 + np.matmul(W[1].T, dz)

    return (np.array([dW0, dW1]), dX)

# Debuging weight matrix in the layer.
# Counting number of dead weights and printing it with the min and 
# max value of the weight in that matrix.     
def debug_matrix(matrix, text = None):
    if DEBUG_MODE == False: return

    if text is not None:
        print (text)

    print ("zeros: %d/%d, non-zero: %d/%d" % 
        (np.sum(matrix == 0), matrix.size, np.sum(matrix != 0), matrix.size))
    print (np.min(matrix), np.max(matrix))
    print ("-" * 20)

# This function is used in the layer where for
# each feature we have 2 linear fucntions 
# Arguments:
#   x:  input matrix
#   W:  matrix of weight matrices (base_layer.weights)
#   spliting_vector: values of splitting point for each feature
#   include_bias: tells if this layer has biases or not.
#   bias_norm: regularizing values of biases.   
# Returns: fucntion's value
def splitting_weight_function_1_point(W, x, split_vector, 
        include_bias = True, bias_norm = 0.01):
    x_0 = np.zeros_like(x)
    x_1 = np.zeros_like(x)
    x_0[x <  split_vector] = x[x <  split_vector]
    x_1[x >= split_vector] = x[x >= split_vector]

    b_0 = np.zeros(x.shape)
    b_0[x < split_vector] = bias_norm
    b_1 = np.zeros(x.shape)
    b_1[x >= split_vector] = bias_norm

    weight_product = np.matmul(W[0], x_0) + np.matmul(W[1], x_1)

    if include_bias:
        bias_product = np.matmul(W[2], b_0) + np.matmul(W[3], b_1)

        debug_matrix(bias_product, "bias")

        return weight_product + bias_product

    return weight_product

# Derivative of the function which is used in the layer
# where for each feature we have 2 linear fucntions 
# Arguments:
#   x:  input matrix
#   W:  matrix of weight matrices (base_layer.weights)
#   dz: derivative of this layer. 
#   spliting_vector: values of splitting point for each feature
#   include_bias: tells if this layer has biases or not.
#   bias_norm: regularizing values of biases.   
# Returns: fucntion's value
def d_splitting_weight_function_1_point(W, x, dz, split_vector = 0, 
        include_bias = True, bias_norm = 0.01):
    x_0 = np.zeros_like(x)
    x_1 = np.zeros_like(x)
    x_0[x <  split_vector] = x[x <  split_vector]
    x_1[x >= split_vector] = x[x >= split_vector]

    b_0 = np.zeros(x.shape)
    b_0[x < split_vector] = bias_norm
    b_1 = np.zeros(x.shape)
    b_1[x >= split_vector] = bias_norm

    dW0 = np.matmul(dz, x_0.T)
    dW1 = np.matmul(dz, x_1.T)

    dX0 = np.matmul(W[0].T, dz)
    dX1 = np.matmul(W[1].T, dz)
    dX  = np.zeros_like(x)
    dX[x <  split_vector] = dX0 [x <  split_vector]
    dX[x >= split_vector] = dX1 [x >= split_vector]

    debug_matrix(dz, "dz")
    debug_matrix(W, "W")

    if include_bias:
        dW2 = np.matmul(dz, b_0.T)
        dW3 = np.matmul(dz, b_0.T)

        debug_matrix(dW2, "dW2")
        debug_matrix(dW3, "dW3")

        return (np.array([dW0, dW1, dW2, dW3]), dX)

    return (np.array([dW0, dW1]), dX)

biased_splt_w_func_1 = partial(splitting_weight_function_1_point, 
    split_vector = np.zeros((3072, 1)), include_bias = True)
d_biased_splt_w_func_1 = partial(d_splitting_weight_function_1_point, 
    split_vector = np.zeros((3072, 1)), include_bias = True)

unbiased_splt_w_func_1 = partial(splitting_weight_function_1_point, 
    split_vector = np.zeros((3072, 1)), include_bias = False)
d_unbiased_splt_w_func_1 = partial(d_splitting_weight_function_1_point, 
    split_vector = np.zeros((3072, 1)), include_bias = False)

"""
    Splitting function is used in the piecewise_layer.
    This function decides for every input example, which
    class it belongs to. Should be written by layer creator
"""
# Send all examples in one class.
def splitting_function_0(X):
    return np.zeros((X.shape[1]))

# Divide examples into two classes: one with mean negative
# other with mean non-negative.
def splitting_function_mean(X):
    return (np.mean(X, axis=0) > 0).astype(int)

# Divide examples by their indexes.  
def splitting_function_even_odd(X):
    return np.arange(X.shape[1]) % 2

# This function is used with the neural network which has a layer with 
# splitting weight function; Function tries to change splitting point 
# of the layer in the trained model to improve the accuracy. Gets one 
# random file from given train files, gets all wrong results and moves 
# the splitting points so that most of the wrong examples end up in new 
# side of splitting point. Runs more epochs to re-train the model.
# Arguments:
#   model: mamba_net model with splitting weight function.
#   file_names: vector of train data file names.
#   n_it: number of iterations.
#   n_epochs: number of epochs in each iteration.
#   n_ranges: number of ranges in splitting points vector.
#   delta: this value decides how much new points should be shifted.
#       greater value means more negative results will be moved into
#       new side of the splitting point.
def learn_piecewise_move_points(model, file_names, n_it=5, n_epochs=5, 
        n_ranges=6, delta=0.01):
    for _ in range(n_it):
        train_file = random.choice(file_names)
        data_x, data_y = get_cifar_dataset(train_file)

        predicted_distr = model.predict(data_x)
        distr_res = np.argmax(predicted_distr, axis=0)

        neg_examples = data_x[:, data_y != distr_res]

        neg_examples_means = np.mean(neg_examples, axis=1)

        splitting_points = np.zeros_like(neg_examples_means)
        range_length = neg_examples.shape[0] // n_ranges

        left_bound = 0
        for i in range(1, n_ranges + 1):
            right_bound = i * range_length

            cur_range_mean = neg_examples_means[left_bound:right_bound].mean()
            cur_range_mean += delta if cur_range_mean > 0 else -delta

            splitting_points[left_bound:right_bound] = cur_range_mean
            left_bound = right_bound
        
        splitting_points = splitting_points.reshape(-1, 1)

        splt_func = partial(splitting_weight_function_1_point,
            split_vector = splitting_points, include_bias = False)
        d_splt_func = partial(d_splitting_weight_function_1_point, 
            split_vector = splitting_points, include_bias = False)

        model.layers[0].weight_function = splt_func
        model.layers[0].d_weight_function = d_splt_func

        print ("-----Splitting Point Shifted!!!-----")

        model.train_from_files(file_names, "test_batch", get_cifar_dataset, 
            learning_rate=0.05, n_epochs=n_epochs, dump_architecture=False,
            stop_len=100, stop_diff=0.001)

def main():

    kobi = MambaNet(12)

    layer1 = BaseLayer(64, "relu", "xavier",
                       (unbiased_splt_w_func_1, d_unbiased_splt_w_func_1),
                       2,
                       False,
                       0.001)
    layer2 = BaseLayer(64, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True,
                       0.001)
    layer3 = BaseLayer(10, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True,
                       0.001)
    kobi.add(layer1)
    kobi.add(layer2)
    kobi.add(layer3)

    kobi.compile(3072, 10)

    file_names = ["data_batch_%s" % (str(ind)) for ind in range(1, 2)]

    kobi.train_from_files(file_names, "test_batch", get_cifar_dataset, 
        learning_rate=0.05, n_epochs=2, dump_architecture=True,
        stop_len=100, stop_diff=0.001)
    
    learn_piecewise_move_points(kobi, file_names, n_epochs=5, n_it=5)

if __name__ == '__main__':
    main()