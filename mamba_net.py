from base_layer import BaseLayer

class MambaNet:
    """
    Main class for NN achitecture management.
    """

    def __init__(self, random_state: int):
        pass

    def add(self, layer):
        # Arguments:
        #   layer: Layer. NN layer to add.

        pass

    def compile(self, input_shape, output_shape):
        # Arguments:
        #   input_shape: shape of input data
        #   output_shape: shape of output data (e.g number of classes)
        # Return: Throws exception on misfitted layer.

        pass

    def train(self,
              x, y,
              validation_data=None,
              validation_split=0.,
              n_epochs=50,
              batch_size=100,
              learning_rate=0.0001):

        pass

    def test(self, x, y):
        pass

    def predict(self, x):
        pass