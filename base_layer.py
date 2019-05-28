from activation_functions import *
from weight_initializators import *
from functools import partial 

class BaseLayer:
    """
    This class is abstraction of neural networks' layers.

    # Arguments:
        n_units: Int. Number of units in current layer.
        activation_function: String. Name of activation function, all the 
            activation functions can be found in ActivationFunctions class.
        weight_functions: Tuple. Callable instance of weight function and its 
            derivative. For example: Wx or W_1*x^2 + W_2*x and their derivate 
            functions.
        n_weight_matrices: Int. Defines order of weight function. Which could
            be 1 (linear), 2 (quadratic), ... (polynomial).
        include_bias: Boolean. Whether we should add b or not. 
    """

    # TODO weight initialization function
    def __init__(self, 
                 n_units: int, 
                 activation_function: str, 
                 initialization_function: str,
                 weight_functions: tuple,
                 n_weight_matrices: int,
                 include_bias: bool,
                 l2_regularization_rate: float = 0.001):
        self.n_units = n_units
        self.activation_function, self.d_activation_function = \
            ActivationFunctions.get(activation_function)
        self.weight_init = \
            WeightInitializators.get(initialization_function)
        self.weight_function, self.d_weight_function = weight_functions
        self.n_weight_matrices = n_weight_matrices
        self.include_bias = include_bias
        self.weights = []
        self.bias = None
        self.l2_regularization_rate = l2_regularization_rate
        self.init_func_name = initialization_function
        self.activ_func_name = activation_function

    def forward_calculation(self, A_prev):
        # Arguments:
        #   A_prev: List. Output from previous layer
        # Return: single value of forward pass

        Wx = self.weight_function(self.weights, A_prev) # M N * N 1 = M 1 
        self.derivative = partial(
            self.d_weight_function,
            W=self.weights, x=A_prev
        )
        self.Z = Wx + self.bias if self.include_bias else Wx

        output = self.activation_function(self.Z)

        return output

    def backward_calculation(self, previous_derivative) -> tuple:
        # Arguments:
        #   previous_derivative: List. Derivative of next layer
        # Return: tuple of dw and dx

        dZ = self.d_activation_function(previous_derivative, self.Z)
        db = np.sum(dZ, axis = 1, keepdims = True) if self.include_bias \
            else np.zeros((dZ.shape[0], 1))

        dW, dA = self.derivative(dz = dZ)

        return (dA, dW, db)

    def update_weights(self, learning_rate, gradient):
        # Arguments:
        #   learning_rate: Double.
        #   gradient: List. dW db 
        self.weights = (self.weights 
                        - learning_rate * gradient[0]
                        - self.l2_regularization_rate * self.weights)
        self.bias = self.bias - learning_rate * gradient[1] \
            if self.include_bias else self.bias

    def initialize_weights(self, prev_layer_shape):
        for i in range(self.n_weight_matrices):
            current_weights = self.weight_init(
                (self.n_units, prev_layer_shape),
                prev_layer_shape
            )
            self.weights.append(current_weights)

        self.weights = np.array(self.weights)

        if self.include_bias:
            self.bias = self.weight_init((self.n_units, 1), prev_layer_shape)