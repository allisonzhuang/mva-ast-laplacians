from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from tslearn.metrics import cdist_dtw
from scipy.ndimage import minimum_filter1d, maximum_filter1d


def create_graph(distance_matrix: np.ndarray, similarity_matrix: np.ndarray, k: Optional[int] = None, y: Optional[np.ndarray] = None):
    if k is not None and k > 0:
        nn = NearestNeighbors(metric="precomputed")
        nn.fit(distance_matrix)

        A = nn.kneighbors_graph(None, k).toarray()
    else:
        n = distance_matrix.shape[0]
        A = np.zeros((n, n))

    if y is not None:
        unique_labels = np.unique(y)

        for label in unique_labels:
            mask = y == label
            idx = np.ix_(mask, mask)

            A[idx] = 1

    np.fill_diagonal(A, 0)
    S = np.where(A == 1, similarity_matrix, 0)

    D = np.diag(S.sum(axis=1))
    L = D - S

    return D, L


def laplacian_score(X, distance_matrix, similarity_matrix, k=None):
    D, L = create_graph(distance_matrix, similarity_matrix, k)

    F_mu = (X.T @ D.diagonal()[:, np.newaxis]) / D.sum()
    F_t = X - F_mu.T  # i.e., F-tilde, take outer product with mean vector for mean matrix

    # calculate vectorized L_r for all r
    num_matrix = F_t.T @ L @ F_t
    num_vector = np.diag(num_matrix)

    denom_matrix = F_t.T @ D @ F_t
    denom_vector = np.diag(denom_matrix)

    L_r_scores = np.divide(num_vector, denom_vector, out=np.zeros_like(num_vector, dtype=float), where=denom_vector != 0)
    return L_r_scores


def euclidean_laplacian_score(X_f, k, t):
    distance_matrix = euclidean_distances(X_f, X_f)
    similarity_matrix = np.exp(-distance_matrix / t)

    return laplacian_score(X_f, distance_matrix, similarity_matrix, k)


def fisher_laplacian_score(X, X_f, y):
    n = X.shape[0]
    distance_matrix = np.zeros((n, n))

    labels, counts = np.unique(y, return_counts=True)
    similarity_matrix = np.zeros((n, n))

    for i, label in enumerate(labels):
        mask = y == label
        idx = np.ix_(mask, mask)

        similarity_matrix[idx] = 1 / counts[i]

    return laplacian_score(X_f, distance_matrix, similarity_matrix)


def dtw_laplacian_score(X, X_f, k, t, distance_matrix_path="dist_matrix.npy"):
    try:
        distance_matrix = np.load(distance_matrix_path)
    except FileNotFoundError:
        distance_matrix = cdist_dtw(X, n_jobs=4)
        np.save(distance_matrix_path, distance_matrix)

    similarity_matrix = np.exp(-distance_matrix / t)

    return laplacian_score(X_f, distance_matrix, similarity_matrix, k)


def get_lb_matrix(X, radius):
    N = X.shape[0]
    width = 2 * radius + 1

    L = minimum_filter1d(X, size=width, axis=1, mode="nearest")
    U = maximum_filter1d(X, size=width, axis=1, mode="nearest")

    lb_matrix = np.zeros((N, N))

    for i in range(N):
        diff_l = np.maximum(0, L - X[i])
        diff_u = np.maximum(0, X[i] - U)

        lb_matrix[i] = np.sqrt(np.sum(diff_l**2 + diff_u**2, axis=1))

    return np.maximum(lb_matrix, lb_matrix.T)


def lb_dtw_laplacian_score(X, X_f, k, t, distance_matrix_path="dist_matrix.npy"):
    try:
        distance_matrix = np.load(distance_matrix_path)
    except FileNotFoundError:
        distance_matrix = get_lb_matrix(X, 30)
        np.save(distance_matrix_path, distance_matrix)

    similarity_matrix = np.exp(-distance_matrix / t)

    return laplacian_score(X_f, distance_matrix, similarity_matrix, k)
