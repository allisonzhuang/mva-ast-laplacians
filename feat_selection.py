import numpy as np
from sklearn.decomposition import PCA


def pca_selector(X):
    pca = PCA()
    pca.fit(X)

    V = pca.components_
    eigenvalues = pca.explained_variance_

    scores = eigenvalues @ (V**2)
    return scores


def fisher_selector(X, y):
    y_unique, counts = np.unique(y, return_counts=True)
    c = len(y_unique)

    class_means = np.zeros((c, X.shape[1]))
    class_vars = np.zeros((c, X.shape[1]))

    global_mean = X.mean(axis=0)

    for i, y_i in enumerate(y_unique):
        X_i = X[y == y_i]
        class_means[i] = X_i.mean(axis=0)
        class_vars[i] = X_i.var(axis=0)

    numerator = (counts[:, None] * (class_means - global_mean) ** 2).sum(axis=0)
    denominator = (counts[:, None] * class_vars).sum(axis=0)

    F = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))
    return F


def filter_features(X, scores, r, return_feat_idx: bool = False):
    r_most_important = np.argsort(scores)[::-1][:r]
    X_sel = X[:, r_most_important]

    if return_feat_idx:
        return X_sel, r_most_important

    return X_sel
