import numpy as np

class WeightInitializators:
    """
    This class provides implementations of various activation functions. Each of 
    these activation functions should be accesible by calling get() with function 
    name as parameter.
    """

    def XavierRelu(shape, prev_layer_shape):
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2 / prev_layer_shape)

    def BengioEtal(shape, prev_layer_shape):
        return 4 * np.random.randn(shape[0], shape[1]) \
                 * np.sqrt(6 / (shape[0] + prev_layer_shape))

    def UniformRandom(shape, prev_layer_shape):
        return 2 * np.random.randn(shape[0], shape[1]) - 1

    registry = {
        "xavier": XavierRelu,
        "bengio": BengioEtal,
        "tupoi": UniformRandom
    }

    @staticmethod
    def get(function_name: str) -> tuple:
        return WeightInitializators.registry.get(function_name, \
            WeightInitializators.UniformRandom)