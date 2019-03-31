from activation_functions import ActivationFunctions

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
                 weight_functions: tuple,
                 weight_function_order: int,
                 include_bias: bool):
        self.number_of_units = number_of_units
        self.activation_function, self.d_activation_function = \
            ActivationFunctions.get(activation_function)
        self.weight_function, self.d_weight_function = weight_functions
        self.weight_function_order = weight_function_order
        self.include_bias = include_bias
        self.weights = None
        self.bias = None

    def forward_calculation(self, A_prev):
        # Arguments:
        #   A_prev: List. Output from previous layer
        # Return: single value of forward pass

        Wx = self.weight_function(self.weights, A_prev)
        self.Z = Wx + self.bias if self.include_bias else Wx

        return self.activation_function(self.Z)

    def backward_calculation(self, previous_derivative) -> tuple:
        # Arguments:
        #   previous_derivative: List. Derivative of next layer
        # Return: tuple of dw and dx

        pass

    def update_weights(self, learning_rate, gradient):
        # Arguments:
        #   learning_rate: Double.
        #   gradient: List. 

        pass

    def initialize_weights(self, prev_layer_shape):
        self.weights = 2 * np.random.rand(prev_layer_shape, self.number_of_units) - 1
        if include_bias:
            self.bias = 2 * np.random.rand(self.number_of_units) - 1