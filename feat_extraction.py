import numpy as np
from itertools import tee
from scipy.stats import skew, kurtosis
from scipy.signal import argrelmax
from scipy.fft import rfft, rfftfreq
from statsmodels.tsa.stattools import acf


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_n_largest(arr, n_largest=3):
    indexes = np.argsort(arr)[-n_largest:][::-1]
    values = arr[indexes]
    return values, indexes


def get_largest_local_max(signal, order=1):
    max_locs = argrelmax(signal, order=order)[0]
    if max_locs.size == 0:
        return 0.0, 0

    values = signal[max_locs]
    largest_idx = np.argmax(values)
    return values[largest_idx], max_locs[largest_idx]


def get_distribution_features(X):
    # X shape: (n_samples, n_timepoints)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    min_val = np.min(X, axis=1, keepdims=True)
    max_val = np.max(X, axis=1, keepdims=True)

    skew_val = skew(X, axis=1).reshape(-1, 1)
    kurt_val = kurtosis(X, axis=1).reshape(-1, 1)

    q25 = np.percentile(X, 25, axis=1, keepdims=True)
    q50 = np.percentile(X, 50, axis=1, keepdims=True)
    q75 = np.percentile(X, 75, axis=1, keepdims=True)

    # Returns shape (n_samples, 9)
    return np.hstack([mean, std, min_val, max_val, skew_val, kurt_val, q25, q50, q75])


def get_fourier_features(X, fs=2000, n_bins=50):
    n_samples, n_points = X.shape

    # Normalize
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_norm = np.divide(X - X_mean, X_std, out=np.zeros_like(X), where=X_std != 0)

    # FFT
    fft_vals = np.abs(rfft(X_norm, axis=1))
    freqs = rfftfreq(n=n_points, d=1.0 / fs)

    freq_bins = np.linspace(0, fs / 2, n_bins + 1)
    features = []

    for f_min, f_max in pairwise(freq_bins):
        mask = (freqs >= f_min) & (freqs < f_max)
        # Sum of squares (power) in bin
        power = np.sum(fft_vals[:, mask] ** 2, axis=1)
        # Log power
        log_power = np.log(np.where(power > 0, power, 1e-10))
        features.append(log_power.reshape(-1, 1))

    return np.hstack(features)


def _get_single_autocorr_features(signal, n_lags, fs):
    # Helper for row-wise processing due to complex peak finding logic
    ac_vals = acf(signal, nlags=n_lags, fft=True)

    # 1. Raw Autocorrelation values
    features = list(ac_vals)

    # 2. Largest Local Max
    # Look at AC excluding lag 0
    val_max, idx_max = get_largest_local_max(ac_vals[1:], order=10)
    lag_hz_max = fs / (idx_max + 1) if (idx_max + 1) > 0 else 0

    # 3. Largest Local Min (using negative peak finding)
    val_min_neg, idx_min = get_largest_local_max(-ac_vals[1:], order=10)
    val_min = -val_min_neg
    lag_hz_min = fs / (idx_min + 1) if (idx_min + 1) > 0 else 0

    features.extend([val_max, lag_hz_max, val_min, lag_hz_min])
    return features


def get_autocorr_features(X, n_lags=60, fs=2000):
    n_samples = X.shape[0]

    # Pre-normalize for ACF consistency
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_norm = np.divide(X - X_mean, X_std, out=np.zeros_like(X), where=X_std != 0)

    # Process row by row
    all_features = []
    for i in range(n_samples):
        row_feats = _get_single_autocorr_features(X_norm[i], n_lags, fs)
        all_features.append(row_feats)

    return np.array(all_features)


def get_features(X, fs=2000, n_fourier_bins=50, n_lags=60, **kwargs):
    """
    X: matrix of time series (N, T)
    Returns: matrix of features (N, F)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    dist_feats = get_distribution_features(X)
    fourier_feats = get_fourier_features(X, fs=fs, n_bins=n_fourier_bins)
    autocorr_feats = get_autocorr_features(X, n_lags=n_lags, fs=fs)

    X_feats = np.hstack([dist_feats, fourier_feats, autocorr_feats])

    return X_feats
