from base_layer import *
from piecewise_layer import *
from datetime import datetime
from os.path import join
import time
import json

MODEL_DIR = "experiments/"

class MambaNet:
    """
    Main class for NN achitecture management.
    """

    def Softmax(z):
        # Arguments:
        #   z: predict matrix (num_classes x num_examples)
        # Returns: softmaxed result (num_classes x num_examples)
        exps = np.exp(z - np.max(z, axis = 0))

        return exps / np.sum(exps, axis = 0)

    def CrossEntropy(y_hat, y):
        # Arguments:
        #   y_hat: predicted classes (num_classes x num_examples) 
        #   y: real classes (num_examples)
        # Returns: loss value ()  
        m = y_hat.shape[1]
        p = y_hat.copy()

        log_likelihood = -np.log(p[y, range(m)] + 1e-5)
        loss = np.sum(log_likelihood) / m

        return loss

    def dCrossEntropy(y_hat, y):
        # Arguments:
        #   y_hat: predicted classes (num_classes x num_examples) 
        #   y: real classes (num_examples)
        # Returns: gradient value
        m = y_hat.shape[1]

        grad = y_hat.copy()
        grad[y, range(m)] -= 1
        # grad = np.sum(grad, axis = 1, keepdims=True)
        grad = grad / m

        return grad

    def __init__(self, random_state: int):
        np.random.seed(random_state)

        self.random_state = random_state
        self.layers = []

    def add(self, layer):
        # Arguments:
        #   layer: Layer. NN layer to add.
        self.layers.append(layer)

    def compile(self, input_shape, output_shape):
        # Arguments:
        #   input_shape: shape of input data
        #   output_shape: shape of output data (e.g number of classes)
        # Return: Throws exception on misfitted layer.  

        prev_layer_shape = input_shape

        for layer in self.layers:
            layer.initialize_weights(prev_layer_shape)
            prev_layer_shape = layer.number_of_units

        if prev_layer_shape != output_shape:
            raise Exception("Output layer's shape does not match!")

    def predict(self, X):
        # Arguments:
        #   X: input matrix (num_feature x num_examples)
        # Returns: predict matrix (num_classes x num_examples)
        A_curr = X

        for layer in self.layers:
            A_curr = layer.forward_calculation(A_curr)

        return MambaNet.Softmax(A_curr)

    def train_from_files(self,
              file_pathes,
              validation_data_path,
              dataset_reader,
              n_epochs=50,
              batch_size=100,
              learning_rate=0.1,
              dump_architecture=False,
              stop_len = 5,
              stop_diff = 0.005):
        
        val_x, val_y = dataset_reader(validation_data_path)

        val_accs = np.array([])
        train_accs = np.array([])

        for epoch in range(n_epochs):
            acc_sum = 0.0
            for file_path in file_pathes:
                current_x, current_y = dataset_reader(file_path)
                self.train(current_x, current_y, 
                    validation_data=None, 
                    validation_split=0, 
                    n_epochs=1, 
                    batch_size=batch_size, 
                    learning_rate=learning_rate, 
                    verbose=0)
                
                acc_sum += self.count_accuracy(current_x, current_y)
            
            train_acc = acc_sum / len(file_pathes)

            val_acc = self.count_accuracy(val_x, val_y)

            print ("Epoch %d, Train-Acc: %.5f, Val-Acc:%.5f" \
                % (epoch, train_acc, val_acc))

            val_accs = np.append(val_accs, val_acc)
            train_accs = np.append(train_accs, train_acc)

            if val_accs.size >= stop_len and np.max(val_accs[-stop_len:]) - np.min(val_accs[-stop_len:]) < stop_diff: break

        if dump_architecture:
            indx = val_accs.argmax()
            self.dump_architecture(
                learning_rate = learning_rate,
                train_acc = train_accs[indx],
                val_acc = val_accs[indx],
                n_epochs = indx + 1)

    def train(self,
              x, y,
              validation_data=None,
              validation_split=0.2,
              n_epochs=50,
              batch_size=100,
              learning_rate=0.1,
              verbose=1,
              dump_architecture=False,
              stop_len = 5,
              stop_diff = 0.05):

        y = np.array(y)
        val_accs = np.array([])

        if not validation_data and validation_split > 0:
            train_x, train_y, val_x, val_y = \
                self._get_validation_data((x, y), validation_split)
        else:
            train_x = x
            train_y = y
            if validation_split == 0:
                val_x, val_y = None, None
            else:
                val_x, val_y = validation_data

        train_acc = self.count_accuracy(train_x, train_y)
        val_acc = self.count_accuracy(val_x, val_y)

        if verbose == 1:
            print ("Epoch %d, Train-Acc: %.5f, Val-Acc:%.5f" % (-1, train_acc, val_acc))

        train_start_time = time.time()
        for epoch in range(n_epochs):
            chunks_x, chunks_y = self._split_data(batch_size, train_x, train_y)
            epoch_start_time = time.time()
            for chunk_x, chunk_y in zip(chunks_x, chunks_y):

                predicted_y = self.predict(chunk_x)

                loss = MambaNet.CrossEntropy(predicted_y, chunk_y)
                loss_gradient = MambaNet.dCrossEntropy(predicted_y, chunk_y)
                
                self._do_backprop(loss_gradient, learning_rate)

            train_acc = self.count_accuracy(train_x, train_y)
            val_acc = self.count_accuracy(val_x, val_y)
            epoch_end_time = time.time()

            if verbose == 1:
                print ("Epoch %d, Train-Acc: %.5f, Val-Acc:%.5f, epoch took time: %.5f, from start: %.5f" % \
                    (epoch, train_acc, val_acc, epoch_end_time - epoch_start_time, epoch_end_time - train_start_time))

            val_accs = np.append(val_accs, val_acc)
            if val_accs.size >= stop_len and np.max(val_accs[-stop_len:]) - np.min(val_accs[-stop_len:]) < stop_diff: break   

        if verbose == 1:
            print("Train finished in: %.5f" % (epoch_end_time - train_start_time))
        
        if dump_architecture:
            self.dump_architecture(learning_rate, train_acc, val_acc, n_epochs)

    def test(self, x, y):
        pass
    
    def count_accuracy(self, x, y):

        if x is None or y is None: return 0.0

        predicted_y = self.predict(x)

        sparse_predicted_y = np.argmax(predicted_y, axis=0)

        acc = np.sum(sparse_predicted_y == y) / len(y)

        return acc

    def dump_architecture(self, learning_rate, train_acc, val_acc, n_epochs):
        def get_layer_parameters(layer):
            def get_base_layer_params(layer):
                return {
                    'n_units': layer.number_of_units,
                    'activation_func': layer.activ_func_name,
                    'initialization_func': layer.init_func_name,
                    'weight_func_order': layer.weight_function_order,
                    'bias': layer.include_bias,
                    'regularization_rate': layer.l2_regularization_rate
                }

            if isinstance(layer, BaseLayer):
                return get_base_layer_params(layer)
            elif isinstance(layer, BasePieceWiseLayer):
                return [get_base_layer_params(layer) for layer in layer.layers]
            else:
                return {}

        parameters = {
            'train_acc': round(train_acc, 7),
            'val_acc': round(val_acc, 7),
            'n_layers': len(self.layers),
            'n_epochs': int(n_epochs),
            'learning rate': learning_rate,
            'random_state': self.random_state
        }

        for index, layer in enumerate(self.layers):
            layer_parameters = get_layer_parameters(layer)
            parameters['layer_%d' % (index)] = layer_parameters
        
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name += '-val%d.json' % int(100 * val_acc)
        file_name = join(MODEL_DIR, file_name)
        with open(file_name, 'w') as o:
            json.dump(parameters, o, indent=4 * ' ')

    def _get_validation_data(self, data, validation_split):
        # TODO come up with better data split algorithm.

        validation_offset = int(len(data[1]) * validation_split)
        val_x = data[0][:, :validation_offset]
        val_y = data[1][:validation_offset]
        train_x = data[0][:, validation_offset:]
        train_y = data[1][validation_offset:]

        return train_x, train_y, val_x, val_y

    def _split_data(self, batch_size, x, y):
        n_iterations = x.shape[1] // batch_size
        chunks_x = np.array_split(x, n_iterations, axis=1)
        chunks_y = np.array_split(y, n_iterations)

        return chunks_x, chunks_y

    def _do_backprop(self, loss_gradient, learning_rate):
        gradient = loss_gradient
        for layer in self.layers[::-1]:
            # print("gradient: ", np.min(gradient), np.max(gradient))
            dA, dW, db = layer.backward_calculation(gradient)
            layer.update_weights(learning_rate, (dW, db))
            # print("weights : ", np.min(layer.weights), np.max(layer.weights))
            gradient = dA