import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    scaler = StandardScaler()
    X_norm_train = scaler.fit_transform(train_features, train_target)
    X_norm_test = scaler.transform(test_features)
    model = SVC(kernel='rbf', C=0.5, class_weight={1: 1, 0: 1})
    model.fit(X_norm_train[:, 3:5], train_target)
    return model.predict(X_norm_test[:, 3:5])
