import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    list = []
    len_fold = num_objects//num_folds
    for i in range(num_folds):
        if i == num_folds - 1:
            fold = np.arange(i*len_fold, num_objects)
            without_fold = np.arange(i*len_fold)
        else:
            fold = np.arange(i*len_fold, (i+1)*len_fold)
            without_fold = np.hstack((np.arange(i*len_fold), np.arange((i+1)*len_fold, num_objects)))
        list.append((without_fold, fold))
    return list


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    res = {}
    for norm in parameters['normalizers']:
        for k in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weights in parameters['weights']:
                    sum_score = 0
                    for fold in folds:
                        norm_train = X[fold[0]]
                        norm_test = X[fold[1]]
                        if norm[0] is not None:
                            norm[0].fit(X[fold[0]])
                            norm_train = norm[0].transform(X[fold[0]])
                            norm_test = norm[0].transform(X[fold[1]])
                        knn = knn_class(n_neighbors=k, metric=metric, weights=weights)
                        knn.fit(norm_train, y[fold[0]])
                        y_pred = knn.predict(norm_test)
                        sum_score += score_function(y[fold[1]], y_pred)
                    res[(norm[1], k, metric, weights)] = float(sum_score)/len(folds)
    return res
