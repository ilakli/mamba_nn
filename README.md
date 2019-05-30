# mamba_nn
mamba_nn stands for Mamba Neural Networks. Why Mamba? Because some of our team members like basketball and one of the best basketball players Kobe Bryant, whose nickname is Black Mamba. Why Neural Networks? Because they are cool, besides whole project is about neural networks. More concretly about trying different modifications of vanilla neural networks. Currently we support three types of modifications: 
1. Polynomyal(quadratic) weight functions. Instead of doing simple, linear W\*x we do W1\*x^2+W2\*x.
1. Parallel weights. In our layer of network architecture we have two sublayers, each of those sublayers is simple vanilla network layer. Our layer decides which sublayer should handle given input.
1. Piecewise weight function. Parameters of layer input are multiplied by different weights.
