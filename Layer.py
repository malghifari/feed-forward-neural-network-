import random
import numpy as np
import math


class Layer:
    def __init__(self, n_neuron, n_input, random_scale=0.01, weights=None):
        self.random_scale = random_scale
        self.n_neuron = n_neuron
        self.n_input = n_input
        if weights is None:
            self.random_weight()
        else:
            self.weights = weights
        self.Delta = np.full((self.n_neuron, self.n_input), 0, dtype='float64')

    def random_weight(self):
        self.weights = self.random_scale * \
            np.random.rand(self.n_neuron, self.n_input)

    def sigmoid(self, z):
        for i in range(len(z)):
            for j in range(len(z[i])):
                try:
                    cur_z = z[i][j]
                    z[i][j] = 0 if -z[i][j] >= 710 else 1 / \
                        (1 + math.exp(-z[i][j]))
                    if z[i][j] < 0:
                        print(cur_z)
                except OverflowError:
                    i = float('inf')
        return z

    def sigmoid_derivative(self, z):
        try:
            a = self.sigmoid(z) * (1 - self.sigmoid(z))
        except Exception:
            a = float('inf')
        return a

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

    def update_weight(self, n_input, learning_rate):
        self.weights -= (self.Delta) * learning_rate
        self.Delta = np.full((self.n_neuron, self.n_input), 0, dtype='float64')
