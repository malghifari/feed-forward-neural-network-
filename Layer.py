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
        for z_list in z:
            for i in z_list:
                i = 0 if -i >= 710 else 1 / (1 + math.exp(-i))
        return z

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feed_forward(self, input):
        self.a = input
        self.z = np.matmul(input, self.weights.T)
        self.output = self.sigmoid(self.z)
        return self.output

    def gradient_descent(self, previous_delta):
        self.delta = previous_delta
        self.Delta = np.matmul(self.delta.T, self.a)

    def compute_delta_output_layer(self, label):
        self.delta = (self.output.T - label).T
        return self.delta

    def compute_delta(self):
        next_delta = np.matmul(self.delta, self.weights) * \
            self.sigmoid_derivative(self.z)
        return next_delta

    def update_weight(self, n_input):
        self.weights -= self.Delta / n_input
        self.Delta = np.full((self.n_neuron, self.n_input), 0, dtype='float64')
