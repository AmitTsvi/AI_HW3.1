import numpy as np
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
    features_numbers = list(np.arange(n_features))
    val = np.inf
    sel = [n_features]
    change = True
    while change:
        change = False
        min_error = np.inf
        min_sel = []
        remaining = [feature for feature in features_numbers if feature not in sel]
        for i in remaining:
            new_sel = sel.copy()
            new_sel.insert(0, i)
            prediction = knn(norm_train_data[:, new_sel], norm_test_data[:, new_sel], k)
            conf_mat = confusion_matrix(norm_test_data[:, -1], prediction)
            error_w = conf_mat[1, 0] + conf_mat[0, 1]
            if error_w < min_error:
                min_sel = new_sel
                min_error = error_w
        if min_error <= val:
            sel = min_sel
            val = min_error
            change = True
    sel = sel[:-1]
    sel.sort()
    print(sel)
