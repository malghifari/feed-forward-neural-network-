import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from FFNN import FFNN

data = arff.loadarff('dataset/weather.arff')

df = pd.DataFrame(data[0])

df['outlook'] = pd.Categorical(df['outlook']).codes
df['windy'] = pd.Categorical(df['windy']).codes
df['play'] = pd.Categorical(df['play']).codes

feature = ['outlook', 'temperature', 'humidity', 'windy']
X = df[feature].to_numpy()
y = df['play'].to_numpy()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
for train_index, test_index in sss.split(X, y):
    training_input, testing_input = X[train_index], X[test_index]
    training_label, testing_label = y[train_index], y[test_index]


ffnn = FFNN(batch_size=10, n_hidden_layers=3, nb_nodes=10,
            learning_rate=0.2, momentum=0.9, epoch=10000, init_epsilon=0.2)

ffnn.fit(training_input, training_label)

ffnn_pred = ffnn.predict(testing_input)
print(classification_report(testing_label, ffnn_pred))