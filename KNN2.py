import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Utils import read_data
from Utils import normalize_data


def predict(sample, data, k):
    distances = [np.linalg.norm(row[:-1]-sample[:-1]) for row in data]
    vote = 0
    for i in range(k):
        min_arg = np.argmin(distances)
        distances[min_arg] = np.inf
        vote += 4*data[min_arg, -1]
    return 1 if vote >= k/2 else 0


def knn(norm_train_data, norm_test_data, k):
    return [predict(x, norm_train_data, k) for x in norm_test_data]


def error_w(ground_truth, prediction):
    conf_mat = confusion_matrix(ground_truth, prediction)
    return 4*conf_mat[1, 0]+conf_mat[0, 1]


if __name__ == '__main__':
    k = 9
    train_data = read_data("train.csv")
    norm_train_data = normalize_data(train_data, train_data)
    test_data = read_data("test.csv")
    norm_test_data = normalize_data(test_data, train_data)
    prediction = knn(norm_train_data, norm_test_data, k)
    conf_mat = confusion_matrix(norm_test_data[:, -1], prediction)
    print("Confusion matrix = ")
    print(conf_mat)
    print("Error_w = " + str(4*conf_mat[1, 0]+conf_mat[0, 1]))
    k_values = [1, 3, 9, 27]
    error_values = [error_w(norm_test_data[:, -1], knn(norm_train_data, norm_test_data, x)) for x in k_values]
    plt.plot(k_values, error_values)
    plt.show()
