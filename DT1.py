# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.metrics import accuracy_score
import random


# ----------- PART A -----------
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

DT1 = tree.DecisionTreeClassifier(criterion="entropy")
DT1 = DT1.fit(X_train, Y_train)
DT1_test = DT1.predict(X_test)
print("DT1:")
print(metrics.confusion_matrix(Y_test, DT1_test))
accuracy_score_DT1 = accuracy_score(Y_test, DT1_test)
#print(accuracy_score_DT1)

# DT3
DT3 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=3)
DT3 = DT3.fit(X_train, Y_train)
DT3_test = DT3.predict(X_test)
print("DT3:")
#print(metrics.confusion_matrix(Y_test, DT3_test))
accuracy_score_DT3 = accuracy_score(Y_test, DT3_test)
print(accuracy_score_DT3)

# DT9
DT9 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=9)
DT9 = DT9.fit(X_train, Y_train)
DT9_test = DT9.predict(X_test)
print("DT9:")
#print(metrics.confusion_matrix(Y_test, DT9_test))
accuracy_score_DT9 = accuracy_score(Y_test, DT9_test)
print(accuracy_score_DT9)

# DT27
DT27 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=27)
DT27 = DT27.fit(X_train, Y_train)
DT27_test = DT27.predict(X_test)
print("DT27:")
#print(metrics.confusion_matrix(Y_test, DT27_test))
accuracy_score_DT27 = accuracy_score(Y_test, DT27_test)
print(accuracy_score_DT27)

# DT27 image
plt.figure(figsize=(40,20))  # customize according to the size of your tree
_ = tree.plot_tree(DT9, feature_names = X_train.columns)
plt.show()


# ----------- PART B -----------
# q9
# print(np.count_nonzero(DT1_test == 1))
DT1_test_05 = DT1_test.copy()
DT1_test_1 = DT1_test.copy()
DT1_test_2 = DT1_test.copy()


def prob_calc (p: float, DT1_test_changed, Y_test):
    for y in range(len(DT1_test_changed)):
        if DT1_test_changed[y] == 0:
            coin = random.random()
            if coin <= p:
                DT1_test_changed[y] = 1
    print("DT1 , " + str(p) + ":")
    print(metrics.confusion_matrix(Y_test, DT1_test_changed))


# 0.05
p = 0.05
prob_calc(p, DT1_test_05, Y_test)

# 0.1
p = 0.1
prob_calc(p, DT1_test_1, Y_test)

# 0.2
p = 0.2
prob_calc(p, DT1_test_2, Y_test)
