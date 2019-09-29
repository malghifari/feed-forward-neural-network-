import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from FFNN import FFNN

df = pd.read_csv('dataset/gender_classification.csv')

df['Favorite Color'] = pd.Categorical(df['Favorite Color']).codes
df['Favorite Music Genre'] = pd.Categorical(df['Favorite Music Genre']).codes
df['Favorite Beverage'] = pd.Categorical(df['Favorite Beverage']).codes
df['Favorite Soft Drink'] = pd.Categorical(df['Favorite Soft Drink']).codes
df['Gender'] = pd.Categorical(df['Gender']).codes

features = ['Favorite Color', 'Favorite Music Genre',
            'Favorite Beverage', 'Favorite Soft Drink']
X = df[features].to_numpy()
y = df['Gender'].to_numpy()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(X, y):
    training_input, testing_input = X[train_index], X[test_index]
    training_label, testing_label = y[train_index], y[test_index]


ffnn = FFNN(batch_size=1000, n_hidden_layers=2, nb_nodes=5,
            learning_rate=1, momentum=0.9, epoch=10)

ffnn.fit(training_input, training_label)

ffnn_pred = ffnn.predict(testing_input)
print(classification_report(testing_label, ffnn_pred))
