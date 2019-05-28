import numpy as np

class WeightInitializators:
    """
    This class provides implementations of various activation functions. Each of 
    these activation functions should be accesible by calling get() with function 
    name as parameter.
    """
    # Xavier initializator. 
    # See: https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
    def XavierRelu(shape, prev_layer_shape):
        return np.random.randn(shape[0], shape[1]) * np.sqrt(2 / prev_layer_shape)
    
    # Bengio initializator.
    # See: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79    
    def BengioEtal(shape, prev_layer_shape):
        return 4 * np.random.randn(shape[0], shape[1]) \
                 * np.sqrt(6 / (shape[0] + prev_layer_shape))

    # Create matrix with values from uiform distribution on (-1,1)               
    def UniformRandom(shape, prev_layer_shape):
        return np.random.uniform(-1,1, size=(shape[0], shape[1]))

    # Map of weights initializators
    registry = {
        "xavier": XavierRelu,
        "bengio": BengioEtal,
        "tupoi": UniformRandom
    }

    @staticmethod
    def get(function_name: str) -> tuple:
        return WeightInitializators.registry.get(function_name, \
            WeightInitializators.UniformRandom)