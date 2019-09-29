import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from FFNN import FFNN

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


ffnn = FFNN(batch_size=100, n_hidden_layers=2, nb_nodes=5,
            learning_rate=0.1, momentum=0.9, epoch=50)

ffnn.fit(training_input, training_label)

ffnn_pred = ffnn.predict(testing_input)
print(ffnn_pred)
print(classification_report(testing_label, ffnn_pred))
