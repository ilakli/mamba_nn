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

    def CrossEntropy(X,y):
        # Arguments:
        #   X: predict softmaxed matrix (num_classes x num_examples) 
        #   y: real classes (num_examples)
        # Returns: loss value ()  
        m = y.shape[1]
        p = X.copy()

        log_likelihood = -np.log(p[y, range(m)])     
        loss = np.sum(log_likelihood) / m
        return loss

    def dCrossEntropy(X,y):
        # Arguments:
        #   X: predict softmaxed matrix (num_classes x num_examples) 
        #   y: real classes (num_examples)
        # Returns: gradient value     
        m = y.shape[1]

        grad = X.copy()
        grad[y, range(m)] -= 1
        grad = np.sum(grad, axis = 1, keepdims=True)
        grad = grad/m
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

        if(prev_layer_shape != output_shape):
            raise Exception("Output layer's shape does not match!")

    def predict(self, X):
        # Arguments:
        #   X: input matrix (num_feature x num_examples)
        # Returns: predict matrix (num_classes x num_examples)
        A_curr = X.copy()

        for layer in self.layers:
            A_curr = layer.forward_calculation(A_curr)

        return Softmax(A_curr)    



    def train(self,
              x, y,
              validation_data=None,
              validation_split=0.,
              n_epochs=50,
              batch_size=100,
              learning_rate=0.0001):

        
        for i in range(n_epochs):
            Y_hat = self.predict(X)




    def test(self, x, y):
        pass

    