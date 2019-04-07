from activation_functions import *
from weight_initializators import *
from functools import partial 

class BaseLayer:
    """
    This class is abstraction of neural networks' layers.

    # Arguments:
        number_of_units: Int. Number of units in current layer.
        activation_function: String. Name of activation function, all the activation
            functions can be found in ActivationFunctions class.
        weight_functions: Tuple. Callable instance of weight function and its derivative.
            For example: Wx or W_1*x^2 + W_2*x and their derivate functions.
        weight_function_order: Int. Defines order of weight function. Which could
            be 1 (linear), 2 (quadratic), ... (polynomial).
        include_bias: Boolean. Whether we should add b or not. 
    """

    # TODO weight initialization function
    def __init__(self, 
                 number_of_units: int, 
                 activation_function: str, 
                 initialization_function: str,
                 weight_functions: tuple,
                 weight_function_order: int,
                 include_bias: bool,
                 l2_regularization_rate: float = 0.001):
        self.number_of_units = number_of_units
        self.activation_function, self.d_activation_function = \
            ActivationFunctions.get(activation_function)
        self.weight_initialization = \
            WeightInitializators.get(initialization_function)
        self.weight_function, self.d_weight_function = weight_functions
        self.weight_function_order = weight_function_order
        self.include_bias = include_bias
        self.weights = []
        self.bias = None
        self.l2_regularization_rate = l2_regularization_rate

    def forward_calculation(self, A_prev):
        # Arguments:
        #   A_prev: List. Output from previous layer
        # Return: single value of forward pass

        Wx = self.weight_function(self.weights, A_prev) # M N * N 1 = M 1 
        self.derivative = partial(self.d_weight_function, W = self.weights, x = A_prev)
        self.Z = Wx + self.bias if self.include_bias else Wx

        return self.activation_function(self.Z)

    def backward_calculation(self, previous_derivative) -> tuple:
        # Arguments:
        #   previous_derivative: List. Derivative of next layer
        # Return: tuple of dw and dx

        dZ = self.d_activation_function(previous_derivative, self.Z)
        db = np.sum(dZ, axis = 1, keepdims = True) if self.include_bias else np.zeros()

        dW, dA = self.derivative(dz = dZ)

        return (dA, dW, db)

    def update_weights(self, learning_rate, gradient):
        # Arguments:
        #   learning_rate: Double.
        #   gradient: List. dW db 
        self.weights = self.weights - learning_rate * gradient[0] - self.l2_regularization_rate * self.weights
        self.bias = self.bias - learning_rate * gradient[1] if self.include_bias else self.bias
 
    def initialize_weights(self, prev_layer_shape):
        for i in range(self.weight_function_order):
            self.weights.append(self.weight_initialization((self.number_of_units, prev_layer_shape), prev_layer_shape))
        self.weights = np.array(self.weights)    
        if self.include_bias:
            self.bias = self.weight_initialization((self.number_of_units, 1), prev_layer_shape)