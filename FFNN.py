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
        self.layer_list.append(Layer(n_neuron=self.nb_nodes, n_input=n_features))
        self.layer_list += [Layer(n_neuron=self.nb_nodes, n_input=self.nb_nodes) for i in range(self.n_hidden_layers - 1)]
        self.layer_list.append(Layer(n_neuron=1, n_input=self.nb_nodes))

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
                else:
                    # delta for hidden layer
                    delta = layer.compute_delta(delta)
                    layer.gradient_descent(delta)
    
            for layer in self.layer_list:
                layer.update_weight(n_input)
            
            i += self.batch_size

    def predict(self, input):
        output = input
        for layer in self.layer_list:
            output = layer.feed_forward(output)
        return [1 if i > 0.5 else 0 for i in output]
        # return output[0]

import pandas as pd

df = pd.read_csv('dataset/Churn_Modelling.csv')

df.head()

features = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

preproc_df = df[features]

preproc_df['Geography'] = pd.Categorical(df['Geography']).codes
# preproc_df['Gender'] = pd.Categorical(df['Gender']).codes

preproc_df.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_df = scaler.fit_transform(preproc_df.to_numpy())

label = df['Exited'].to_numpy()

label

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
X = scaled_df
y = label
for train_index, test_index in sss.split(X, y):
    training_input, testing_input = X[train_index], X[test_index]
    training_label, testing_label = y[train_index], y[test_index]


from FFNN import FFNN

ffnn = FFNN(batch_size=5, n_hidden_layers=2, nb_nodes=4, learning_rate=0.1, momentum=0.9, epoch=1)

ffnn.fit(training_input, training_label)

ffnn_pred = ffnn.predict(testing_input)

print(classification_report(testing_label, ffnn_pred))


# import pandas as pd
# from scipy.io import arff

# data = arff.loadarff('dataset/weather.arff')

# df = pd.DataFrame(data[0])

# df.head()


# df['outlook'] = pd.Categorical(df['outlook']).codes
# df['windy'] = pd.Categorical(df['windy']).codes
# df['play'] = pd.Categorical(df['play']).codes

# df.head()


# feature = ['outlook', 'temperature', 'humidity', 'windy']
# X = df[feature].to_numpy()
# y = df['play'].to_numpy()




# ffnn = FFNN()

# ffnn.fit(X, y)

# print(ffnn.predict([1, 70.0, 96.0, 0]))

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(learning_rate_init=0.1, , solver='sgd', batch_size=10, momentum=0.9)