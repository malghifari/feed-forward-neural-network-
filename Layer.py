import random
import numpy as np
import math


class Layer:
    def __init__(self, n_neuron, n_input, init_epsilon=2, weights=None):
        self.init_epsilon = init_epsilon
        self.n_neuron = n_neuron
        self.n_input = n_input
        if weights is None:
            self.random_weight()
        else:
            self.weights = weights
        self.delta_w = np.full(
            (self.n_neuron, 1 + self.n_input), 0, dtype='float64')
        self.prev_delta_w = np.full(
            (self.n_neuron, 1 + self.n_input), 0, dtype='float64')

    def random_weight(self):
        self.weights = np.ones((self.n_neuron, 1))
        random = 2 * self.init_epsilon * \
            np.random.rand(self.n_neuron, self.n_input) - self.init_epsilon
        self.weights = np.append(self.weights, random, axis=1)

    def sigmoid(self, z):
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                try:
                    z[i][j] = 0 if -z[i][j] >= 709 else 1 / \
                        (1 + math.exp(-z[i][j]))
                except OverflowError:
                    z[i][j] = math.nan
        return z

    def sigmoid_derivative(self, z):
        try:
            output = self.sigmoid(z) * (1 - self.sigmoid(z))
        except Exception:
            output = math.nan
        return output

    def feed_forward(self, input):
        self.input = np.append(np.ones((input.shape[0], 1)), input, axis=1)
        self.z = np.matmul(self.input, self.weights.T)
        self.output = self.sigmoid(self.z)
        return self.output

    def gradient_descent(self, previous_delta, n_input, learning_rate, momentum):
        self.delta = previous_delta
        self.prev_delta_w = self.delta_w
        self.delta_w = (learning_rate * (np.matmul(self.delta.T, self.input) /
                                         n_input)) + (momentum * self.prev_delta_w)

    def compute_delta_output_layer(self, label):
        self.delta = (self.output.T - label).T
        return self.delta

    def compute_delta(self, prev_delta, prev_weights):
        next_delta = np.matmul(prev_delta, prev_weights) * \
            self.sigmoid_derivative(
                np.append(np.ones((self.z.shape[0], 1)), self.z, axis=1))
        return next_delta[:, 1:]

    def update_weight(self):
        self.weights -= self.delta_w
