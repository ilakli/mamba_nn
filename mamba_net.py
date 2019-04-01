from base_layer import *

class MambaNet:
    """
    Main class for NN achitecture management.
    """

    def Softmax(z):
        # Arguments:
        #   z: predict matrix (num_classes x num_examples)
        # Returns: softmaxed result (num_classes x num_examples)
        exps = np.exp(z - np.max(z))

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
              validation_split=0.,
              n_epochs=50,
              batch_size=100,
              learning_rate=0.01):

        y = np.array(y)

        for epoch in range(n_epochs):
            chunks_x, chunks_y = self._split_data(batch_size, x, y)

            for chunk_x, chunk_y in zip(chunks_x, chunks_y):

                predicted_y = self.predict(chunk_x)

                loss = MambaNet.CrossEntropy(predicted_y, chunk_y)
                loss_gradient = MambaNet.dCrossEntropy(predicted_y, chunk_y)
                self._do_backprop(loss_gradient, learning_rate)
            
            current_accuracy = self.count_accuracy(x, y)

            print ("Epoch %d, Acc: %.5f" % (epoch, current_accuracy))

    def test(self, x, y):
        pass
    
    def count_accuracy(self, x, y):
        predicted_y = self.predict(x)

        sparse_predicted_y = np.argmax(predicted_y, axis=0)

        acc = np.sum(sparse_predicted_y == y) / len(y)

        return acc

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