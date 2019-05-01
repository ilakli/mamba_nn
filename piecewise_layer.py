from activation_functions import *
from weight_initializators import *
from functools import partial 

class BasePieceWiseLayer:
    """
    This is neural network layer, with piece wise function.

    # Arguments:
        layers: List[BaseLayer].
        splitting_function: This is function which splits data for BaseLayers
    """

    # TODO check if number_of_units is equal for layers
    def __init__(self,
                 layers: list,
                 splitting_function):
        self.layers = layers
        self.splitting_function = splitting_function
        self.number_of_units = layers[0].number_of_units

    def forward_calculation(self, A_prev):
        layer_indexes = self.splitting_function(A_prev)

        return_array = np.zeros((self.number_of_units, A_prev.shape[1]))
        for index, layer in enumerate(self.layers):
            current_data_ind = np.where(layer_indexes == index)[0]

            if len(current_data_ind) == 0: continue

            current_data = A_prev[:, current_data_ind]
            current_data_output = layer.forward_calculation(current_data)

            return_array[:, current_data_ind] = current_data_output
        
        self.layer_indexes = layer_indexes

        return return_array

    def backward_calculation(self, previous_derivative) -> tuple:
        layer_indexes = self.layer_indexes

        dA = np.zeros((self.prev_layer_shape, previous_derivative.shape[1]))
        dW, db = [], []
        for index, layer in enumerate(self.layers):
            current_derivative_ind = np.where(layer_indexes == index)[0]

            if len(current_derivative_ind) == 0:
                dW.append([])
                db.append([])
                continue

            current_derivative = previous_derivative[:, current_derivative_ind]
            current_derivative_gradient = layer.backward_calculation(current_derivative)

            current_dA, current_dW, current_db = current_derivative_gradient

            dW.append(current_dW)
            db.append(current_db)

            dA[:, current_derivative_ind] = current_dA

        return (dA, dW, db)

    def update_weights(self, learning_rate, gradient):
        dWs, dbs = gradient

        for index, (dW, db) in enumerate(zip(dWs, dbs)):
            if len(dW) == 0: continue

            self.layers[index].update_weights(learning_rate, (dW, db))
 
    def initialize_weights(self, prev_layer_shape):
        self.prev_layer_shape = prev_layer_shape
        for layer in self.layers:
            layer.initialize_weights(prev_layer_shape)