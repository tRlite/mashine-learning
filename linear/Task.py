import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.uniq = X.apply(lambda x: sorted(np.unique(x)))

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        res = np.empty((X.shape[0], sum(self.uniq.apply(len))), dtype=self.dtype)
        i = 0
        for col in X.columns:
            for new_col in self.uniq[col]:
                res[:, i] = (np.array(X[col]) == new_col).T
                i += 1
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.succ_col_dct = {}
        self.count_col_dct = {}

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        copy = X.copy()
        copy['target'] = Y

        def f(col):
            gr = copy.groupby(col)
            self.count_col_dct[col] = dict(gr.size()/len(X))
            self.succ_col_dct[col] = dict(gr['target'].mean())
        vf = np.vectorize(f)
        vf(X.columns)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        res = np.empty((X.shape[0], 3*X.shape[1]), dtype=self.dtype)
        i = 0
        for col in X.columns:
            res[:, i] = X[col].apply(lambda x: self.succ_col_dct[col][x])
            res[:, i + 1] = X[col].apply(lambda x: self.count_col_dct[col][x])
            res[:, i + 2] = (res[:, i] + a)/(res[:, i + 1] + b)
            i += 3
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.counters = {}

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        for fold, train in group_k_fold(len(X), self.n_folds, seed):
            self.counters[frozenset(fold)] = SimpleCounterEncoder()
            self.counters[frozenset(fold)].fit(X.iloc[train], Y.iloc[train])

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        res = np.empty((X.shape[0], 3 * X.shape[1]), dtype=self.dtype)
        for fold in self.counters:
            res[np.array(list(fold))] = self.counters[fold].transform(X.iloc[np.array(list(fold))], a, b)
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    _, idx1 = sorted(np.unique(x, return_counts=True), key=lambda x: x[0])
    _, idx2 = sorted(np.unique(x[y == 1], return_counts=True), key=lambda x: x[0])
    return idx2/idx1
