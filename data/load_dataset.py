import numpy as np
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification


def _parse_stocks(data, period, pred_days):
    N = data.size
    series = []
    outcomes = []

    for i in range(0, N, period):
        if i + period + pred_days >= N:
            continue

        series.append(data[i : i + period])

        outcome = np.sign(data[i + period + pred_days] - data[i + period])
        outcomes.append(outcome)

    X = np.array(series)
    y = np.array(outcomes)

    return X, y


def load_dataset(dataset_name: str, split_data=False, **kwargs):
    dataset_name = dataset_name.lower()

    if dataset_name == "heartbeat":
        X, y = load_classification("AbnormalHeartbeat")
        X = X.reshape(X.shape[0], X.shape[2])  # If we can handle multi-dimensional, remove this
    elif dataset_name == "japanese":
        X, y = load_classification("JapaneseVowels")
    elif dataset_name == "microsoft":
        data = np.genfromtxt("MSFT.csv", delimiter=",", skip_header=1, usecols=[1])
        X, y = _parse_stocks(data, kwargs.get("period", 20), kwargs.get("pred_days", 3))
    elif dataset_name == "amazon":
        data = np.genfromtxt("AMZN.csv", delimiter=",", skip_header=1, usecols=[1])
        X, y = _parse_stocks(data, kwargs.get("period", 20), kwargs.get("pred_days", 3))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")

    if split_data:
        return train_test_split(X, y, test_size=0.2)
    return X, y
