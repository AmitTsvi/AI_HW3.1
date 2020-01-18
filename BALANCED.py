# imports
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.metrics import accuracy_score

# ----------- PART B -----------
# q8
# |T|
train_data = pd.read_csv("train.csv")
sum_T = train_data[['Outcome']].eq(1).sum()
train_data_filtered_T = train_data[train_data['Outcome'] == 1]
train_data_filtered_F = train_data[train_data['Outcome'] == 0]
train_data_filtered_F_T = train_data_filtered_F.head(sum_T[0])
train_data_balanced = pd.concat([train_data_filtered_T, train_data_filtered_F_T]).sort_index()
#print(train_data)

# Read data from file 'train.csv' and check it on 'test.csv'
X_train = train_data_balanced[train_data_balanced.columns.difference(['Outcome'])]
#print(X_train)
Y_train = train_data_balanced[['Outcome']]
#print(Y_train)

test_data = pd.read_csv("test.csv")
X_test = test_data[test_data.columns.difference(['Outcome'])]
Y_test = test_data[['Outcome']]

BALANCED = tree.DecisionTreeClassifier(criterion="entropy")
BALANCED = BALANCED.fit(X_train, Y_train)
BALANCED_test = BALANCED.predict(X_test)
print("BALANCED:")
print(metrics.confusion_matrix(Y_test, BALANCED_test))
#accuracy_score_BALANCED = accuracy_score(Y_test, BALANCED_test)
#print(accuracy_score_BALANCED)



