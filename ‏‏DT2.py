# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.metrics import accuracy_score
import random


# ----------- PART B -----------
# q11
# Read data from file 'train.csv' and check it on 'test.csv'
train_data = pd.read_csv("train.csv")
#print(train_data)
X_train = train_data[train_data.columns.difference(['Outcome'])]
#print(X_train)
Y_train = train_data[['Outcome']]
#print(Y_train)

test_data = pd.read_csv("test.csv")
X_test = test_data[test_data.columns.difference(['Outcome'])]
Y_test = test_data[['Outcome']]

# DT9
DT9 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=9, class_weight={0:4,1:1})
DT9 = DT9.fit(X_train, Y_train)
DT9_test = DT9.predict(X_test)
print("DT9:")
print(metrics.confusion_matrix(Y_test, DT9_test))
accuracy_score_DT9 = accuracy_score(Y_test, DT9_test)
print(accuracy_score_DT9)



