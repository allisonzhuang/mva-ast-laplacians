import numpy as np
from sklearn.decomposition import DictionaryLearning


def denoise_data(signals, n_atoms=50, sparsity=0.0):
    dict_learning = DictionaryLearning(n_atoms, alpha=sparsity, max_iter=100)

    V = dict_learning.fit_transform(signals)
    D = dict_learning.components_

    X_hat = V @ D

    return X_hat


def _parse_stocks(data, period=60, pred_days=7, shift_days=1, **kwargs):
    N = data.size
    series = []
    outcomes = []

    for i in range(0, N - period - pred_days, shift_days):
        series.append(data[i : i + period])

        outcome = np.sign(data[i + period + pred_days] - data[i + period])
        outcomes.append(outcome)

    X = np.array(series)
    y = np.array(outcomes)

    return X, y


def load_dataset(dataset_name: str, denoise=True, raw=False, **kwargs):
    dataset_name = dataset_name.lower()

    if dataset_name == "heartbeat":
        from aeon.datasets import load_classification
        X, y = load_classification("AbnormalHeartbeat")

        if not raw:
            X = X.reshape(
                X.shape[0], X.shape[2]
            )  # If we can handle multi-dimensional, remove this

        if denoise:
            X = denoise_data(X, n_atoms=100, sparsity=0.5)
    elif dataset_name == "microsoft":
        data = np.genfromtxt(
            "data/MSFT.csv", delimiter=",", skip_header=1, usecols=[1]
        )

        X, y = data, None

        if not raw:
            X, y = _parse_stocks(data, **kwargs)

        if denoise:
            X = denoise_data(X)

    elif dataset_name == "amazon":
        data = np.genfromtxt(
            "data/AMZN.csv", delimiter=",", skip_header=1, usecols=[1]
        )

        X, y = data, None

        if not raw:
            X, y = _parse_stocks(data, **kwargs)

        if denoise:
            X = denoise_data(X)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")

    return X, y
