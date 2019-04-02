import numpy as np

class ActivationFunctions:
    """
    This class provides implementations of activation functions (and their derivatives). 
    Each of those activation functions should be accesible by calling get() with function name
    """

    def Relu(z):
        return np.maximum(z, 0)

    def dRelu(dA, z):
        d_z = dA.copy()
        d_z[z <= 0] = 0
        return d_z

    def Sigmoid(z):
        return 1. / (1 + np.exp(-z))

    def dSigmoid(dA, z):
        sigmoid_z = ActivationFunctions.Sigmoid(z)
        return dA * sigmoid_z * (1 - sigmoid_z)

    registry = {
        "relu": (Relu, dRelu),
        "sigmoid": (Sigmoid, dSigmoid)
    }

    @staticmethod
    def get(function_name: str) -> tuple:
        if function_name in ActivationFunctions.registry:
            return ActivationFunctions.registry[function_name]
        else:
            raise Exception("No such activation function found")