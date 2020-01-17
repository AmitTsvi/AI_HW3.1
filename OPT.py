import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from Utils import read_data
from Utils import normalize_data
from KNN1 import knn


if __name__ == '__main__':
    k = 9
    train_data = read_data("train.csv")
    norm_train_data = normalize_data(train_data, train_data)
    test_data = read_data("test.csv")
    norm_test_data = normalize_data(test_data, train_data)
    n_features = train_data.shape[1] - 1
    min_val = np.inf
    min_sel = []
    for sel in itertools.product([0, 1], repeat=n_features):
        feature_sel = np.nonzero(sel)[0]
        if feature_sel.size != 0:
            feature_sel = np.append(feature_sel, [n_features])
            prediction = knn(norm_train_data[:, feature_sel], norm_test_data[:, feature_sel], k)
            conf_mat = confusion_matrix(norm_test_data[:, -1], prediction)
            error_w = 4*conf_mat[1, 0]+conf_mat[0, 1]
            if error_w < min_val:
                min_sel = feature_sel[:-1]
                min_val = error_w
    print(min_sel)




