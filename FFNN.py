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
        self.layer_list.append(Layer(n_neuron=self.nb_nodes, n_input=n_features))
        self.layer_list = [Layer(n_neuron=self.nb_nodes, n_input=self.nb_nodes) for i in range(self.n_hidden_layers - 1)]
        self.layer_list.append(Layer(n_neuron=1, n_input=n_features))

        pred = self.predict(X)
        i = 0
        while (i < n_input):

            label = y[i]

            for index, layer in enumerate(reversed(self.layer_list)):
                if (index == 0):
                    # delta for output layer
                    delta = layer.compute_delta_output_layer(y)
                    
                else:
                    # delta for hidden layer
                    layer.gradient_descent(delta)
                    delta = layer.compute_delta(delta)
            
            i += 1

        for layer in self.layer_list:
            layer.update_weight(n_input)

            

    def predict(self, input):
        output = input
        for layer in self.layer_list:
            output = layer.feed_forward(output)
        # return 1 if output[0] > 0.5 else 0
        return output[0]
    
    def cost(self, X, y):
        m = X.shape[0];
        return 1 / (2 * m) * sum(([self.predict(X[i]) for i in range(m)] - y) ** 2)


import pandas as pd
from scipy.io import arff

data = arff.loadarff('dataset/weather.arff')

df = pd.DataFrame(data[0])

df.head()


df['outlook'] = pd.Categorical(df['outlook']).codes
df['windy'] = pd.Categorical(df['windy']).codes
df['play'] = pd.Categorical(df['play']).codes

df.head()


feature = ['outlook', 'temperature', 'humidity', 'windy']
X = df[feature].to_numpy()
y = df['play'].to_numpy()




ffnn = FFNN()

ffnn.fit(X, y)

print(ffnn.predict([1, 70.0, 96.0, 0]))