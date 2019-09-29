import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from Layer import Layer
# from Layer import Layer


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
            i = 0
            while (i < n_input):
                print('===== Index {} ====='.format(i))

                # print(X[index:index + self.batch_size])
                pred = self.predict(X[i:i + self.batch_size])
                label = y[i:i + self.batch_size]

                for index, layer in enumerate(reversed(self.layer_list)):
                    if (index == 0):
                        # delta for output layer
                        delta = layer.compute_delta_output_layer(label)
                        layer.gradient_descent(delta)
                        delta = layer.compute_delta()
                    elif (index == len(self.layer_list) - 1):
                        layer.gradient_descent(delta)
                    else:
                        # delta for hidden layer
                        layer.gradient_descent(delta)
                        delta = layer.compute_delta()

                for layer in self.layer_list:
                    layer.update_weight(n_input)

                i += self.batch_size

    def predict(self, input):
        output = input
        for layer in self.layer_list:
            output = layer.feed_forward(output)
        # return [1 if i >= 0 else 0 for i in output]
        return output


import pandas as pd

df = pd.read_csv('dataset/breast-cancer.csv')

df['Class'] = pd.Categorical(df['Class']).codes
no_miss_val_df = df.copy()
no_miss_val_df['Bare Nuclei'] = df['Bare Nuclei'].replace('?', 'nan')
no_miss_val_df['Bare Nuclei'] = no_miss_val_df['Bare Nuclei'].astype(float)
no_miss_val_df['Bare Nuclei'] = no_miss_val_df['Bare Nuclei'].fillna(no_miss_val_df['Bare Nuclei'].median())

features = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

X = no_miss_val_df[features].to_numpy()
y = no_miss_val_df['Class'].to_numpy()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(X, y):
    training_input, testing_input = X[train_index], X[test_index]
    training_label, testing_label = y[train_index], y[test_index]


ffnn = FFNN(batch_size=1000, n_hidden_layers=2, nb_nodes=5,
            learning_rate=0.1, momentum=0.9, epoch=2)

ffnn.fit(training_input, training_label)

ffnn_pred = ffnn.predict(testing_input)
print(ffnn_pred)
print(classification_report(testing_label, ffnn_pred))
