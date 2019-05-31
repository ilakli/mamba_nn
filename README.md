# mamba_nn
mamba_nn stands for Mamba Neural Networks. Why Mamba? Because some of our team members like basketball and one of the best basketball players Kobe Bryant, whose nickname is Black Mamba. Why Neural Networks? Because they are cool, besides whole project is about neural networks. More concretly about trying different modifications of vanilla neural networks. Currently we support three types of modifications: 
1. Polynomyal(quadratic) weight functions. Instead of doing simple, linear W\*x we do W1\*x^2+W2\*x.
1. Parallel weights. In our layer of network architecture we have two sublayers, each of those sublayers is simple vanilla network layer. Our layer decides which sublayer should handle given input.
1. Piecewise weight function. Parameters of layer input are multiplied by different weights.

# Installation
Installation is pretty straightforward you just pull this project on your computer. At this moment we support two types of datasets `CIFAR-10` and `FASHION-MNIST`, if you would like to run some experiments on `FASHION-MNIST` you need to unzip `fashionmnist.zip`. To succesfully run the project you need to have installed `Python 3` with `numpy` package.

# Usage
Driver file for this project is `main.py`. This is where one should create neural network architecture and run experiments.
To successfully train a network you need to do couple of things:
* Create `MambaNet` object `model = MambaNet(24)` (we include random seed so that results stay consistent).
* Create layers and add them to the model. At this moment we support two types of layers: `BaseLayer` and `BasePieceWiseLayer` which can be seen in `base_layer.py` and `piecewise_layer.py` files. Different types of activation and weight initialization functions can be seen in `activation_functions.py` and `weight_initializators.py`. For example let's create couple of `BaseLayer` examples.
```python
    layer1 = BaseLayer(64, "relu", "xavier",
                       (unbiased_splt_w_func_1, d_unbiased_splt_w_func_1),
                       2,
                       False,
                       0.001)
    layer2 = BaseLayer(64, "sigmoid", "bengio",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True,
                       0.001)
    layer3 = BaseLayer(10, "relu", "xavier",
                       (linear_weight_function, d_linear_weight_function),
                       1,
                       True,
                       0.001)
```
* Add created layers to the model
```python
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
```
* Compile model using input and output size, for `CIFAR-10` dataset this would look like this: `model.compile(3072, 10)` (there are 3072 features for each image and 10 possible labels). During compilation `MambaNet` runs weight initialization for each layer and checks if output of each layer can be fed into next layer.
* `CIFAR-10` dataset appeared to be too large for our computers (couldn't fit in RAM) so we created `train_from_files` function in `MambaNet` which is useful if you face the poor RAM problem. Last step is to run for training, which can be done like this
```python
    file_names = ["data_batch_%s" % (str(ind)) for ind in range(1, 6)]

    model.train_from_files(file_names, "test_batch", get_cifar_dataset, 
        learning_rate=0.05, n_epochs=50, dump_architecture=True,
        stop_len=100, stop_diff=0.001)
```
