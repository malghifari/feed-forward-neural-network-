import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from Layer import Layer


class FFNN:
    def __init__(self, batch_size=1, n_hidden_layers=1, nb_nodes=2, learning_rate=0.1, momentum=0.1, epoch=1):
        self.batch_size = batch_size
        self.n_hidden_layers = n_hidden_layers
        self.nb_nodes = nb_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epoch = epoch

    def fit(self, X, y):
        # Get number of features
        n_input = X.shape[0]
        n_features = X.shape[1]
        # Create list of layers (while initiating random weights)
        self.layer_list = []
        self.layer_list.append(
            Layer(n_neuron=self.nb_nodes, n_input=n_features))
        self.layer_list += [Layer(n_neuron=self.nb_nodes, n_input=self.nb_nodes)
                            for i in range(self.n_hidden_layers - 1)]
        self.layer_list.append(Layer(n_neuron=1, n_input=self.nb_nodes))

        for epoch in range(self.epoch):
            print('===== Epoch {} ====='.format(epoch))
            i = 0
            while (i < n_input):

                pred = self.predict(X[i:i + self.batch_size])
                label = y[i:i + self.batch_size]

                for index, layer in enumerate(reversed(self.layer_list)):
                    if (index == 0):
                        # delta for output layer
                        delta = layer.compute_delta_output_layer(label)
                        layer.gradient_descent(
                            delta, self.batch_size, self.learning_rate)
                    elif (index == len(self.layer_list) - 1):
                        layer.gradient_descent(
                            delta, self.batch_size, self.learning_rate)
                    else:
                        # delta for hidden layer
                        delta = layer.compute_delta(delta, prev_weights)
                        layer.gradient_descent(
                            delta, n_input, self.learning_rate)
                    prev_weights = layer.weights
                for layer in self.layer_list:
                    layer.update_weight()

                i += self.batch_size

    def predict(self, input):
        output = input
        for layer in self.layer_list:
            output = layer.feed_forward(output)
        return [1 if i >= 0.5 else 0 for i in output]
