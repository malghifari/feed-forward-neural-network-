import random
import numpy as np
import math

class Layer:
    def __init__(self, n_neuron, n_input, min_ran=-10, max_ran=10, weights=None):
        self.min_ran = min_ran
        self.max_ran = max_ran
        self.n_neuron = n_neuron
        self.n_input = n_input
        if weights is None:
            self.random_weight()
        else:
            self.weights = weights
    
    def random_weight(self):
        self.weights = np.ndarray(dtype=float, shape=(self.n_neuron, self.n_input))
        for i in range(self.n_neuron):
            for j in range(self.n_input):
                self.weights[i][j] = random.randint(self.min_ran, self.max_ran)

    def feed_forward(self, input):
        z = np.matmul(self.weights, input)
        output = np.ndarray(dtype=float, shape=(len(z)))
        for i in range(len(z)):
            output[i] = self.sigmoid(z[i])
        return output

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))