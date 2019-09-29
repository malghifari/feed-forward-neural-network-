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
        self.Delta = np.full((self.n_neuron, self.n_input), 0, dtype='float64')

    def random_weight(self):
        self.weights = 2 * self.random_scale * \
            np.random.rand(self.n_neuron, self.n_input) - self.random_scale

    def sigmoid(self, z):
        if (-z >= 710): # Avoid math.exp overflow
            return 0
        return 1 / (1 + math.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feed_forward(self, input):
        sigmoid_func = np.vectorize(self.sigmoid)
        print(input.shape, self.weights.T.shape)
        
        self.z = np.matmul(input, self.weights.T)
        self.a = sigmoid_func(self.z)
        return self.a

    def gradient_descent(self, previous_delta):
        print(self.a.shape, previous_delta.shape)
        print('Delta', self.Delta.shape)
        self.Delta += self.a * previous_delta

    def compute_delta_output_layer(self, label):
        print(self.a, label)
        print('aa', (self.a.T -label).shape)
        return (self.a.T - label).T

    def compute_delta(self, previous_delta):
        print(self.weights.T.shape, previous_delta.shape, self.sigmoid_derivative(self.z).shape)
        return np.matmul(self.weights.T, previous_delta) * self.sigmoid_derivative(self.z)

    def update_weight(self, n_input):
        self.weights -= self.Delta / n_input


# layer = Layer(n_neuron=2, n_input=4)
# print(layer.weights)
# print(layer.feed_forward([1,2,3,4]))
