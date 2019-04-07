from base_layer import *
import time

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

        log_likelihood = -np.log(p[y, range(m)])
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
        A_curr = X.copy()

        for layer in self.layers:
            A_curr = layer.forward_calculation(A_curr)

        return MambaNet.Softmax(A_curr)

    def train(self,
              x, y,
              validation_data=None,
              validation_split=0.2,
              n_epochs=50,
              batch_size=100,
              learning_rate=0.1,
              regularization_rate = 0.01):

        y = np.array(y)

        if not validation_data:
            train_x, train_y, val_x, val_y = \
                self._get_validation_data((x, y), validation_split)
        else:
            train_x = np.array(x)
            train_y = np.array(y)
            val_x, val_y = validation_data

        train_acc = self.count_accuracy(train_x, train_y)
        val_acc = self.count_accuracy(val_x, val_y)

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
            print ("Epoch %d, Train-Acc: %.5f, Val-Acc:%.5f, epoch took time: %.5f, from start: %.5f" % \
                 (epoch, train_acc, val_acc, epoch_end_time - epoch_start_time, epoch_end_time - train_start_time))
        print("Train finished in: %.5f" % (epoch_end_time - train_start_time))

    def test(self, x, y):
        pass
    
    def count_accuracy(self, x, y):
        predicted_y = self.predict(x)

        sparse_predicted_y = np.argmax(predicted_y, axis=0)

        acc = np.sum(sparse_predicted_y == y) / len(y)

        return acc

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
            dA, dW, db = layer.backward_calculation(gradient)
            layer.update_weights(learning_rate, (dW, db))
            gradient = dA