import random
import numpy as np
import math

class Layer:
    def __init__(self, n_neuron, n_input, random_scale=10, weights=None):
        self.random_scale = random_scale
        self.n_neuron = n_neuron
        self.n_input = n_input
        if weights is None:
            self.random_weight()
        else:
            self.weights = weights
    
    def random_weight(self):
        self.weights = 2 * self.random_scale * np.random.rand(self.n_neuron, self.n_input) - self.random_scale

    def feed_forward(self, input):
        z = np.matmul(self.weights, input)
        sigmoid_func = np.vectorize(self.sigmoid)
        return sigmoid_func(z)
    
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

# layer = Layer(n_neuron=2, n_input=4)
# print(layer.weights)
# print(layer.feed_forward([1,2,3,4]))