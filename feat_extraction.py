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

    skew_val = skew(X, axis=1, nan_policy="propagate")
    skew_val = np.nan_to_num(skew_val).reshape(-1, 1)

    kurt_val = kurtosis(X, axis=1, nan_policy="propagate")
    kurt_val = np.nan_to_num(kurt_val).reshape(-1, 1)

    q25 = np.percentile(X, 25, axis=1, keepdims=True)
    q50 = np.percentile(X, 50, axis=1, keepdims=True)
    q75 = np.percentile(X, 75, axis=1, keepdims=True)

    return np.hstack(
        [mean, std, min_val, max_val, skew_val, kurt_val, q25, q50, q75]
    )


def get_fourier_features(X, fs=2000, n_bins=50):
    n_samples, n_points = X.shape

    # Normalize
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)

    X_norm = np.divide(
        X - X_mean, X_std, out=np.zeros_like(X), where=X_std > 1e-10
    )

    fft_vals = np.abs(rfft(X_norm, axis=1))
    freqs = rfftfreq(n=n_points, d=1.0 / fs)

    freq_bins = np.linspace(0, fs / 2, n_bins + 1)
    features = []

    for f_min, f_max in pairwise(freq_bins):
        mask = (freqs >= f_min) & (freqs < f_max)
        power = np.sum(fft_vals[:, mask] ** 2, axis=1)
        # Log power (already handled safely in original, but good to keep)
        log_power = np.log(np.where(power > 0, power, 1e-10))
        features.append(log_power.reshape(-1, 1))

    return np.hstack(features)


def _get_single_autocorr_features(signal, n_lags, fs):
    if np.allclose(signal, 0) or np.std(signal) < 1e-10:
        return [0.0] * (n_lags + 1 + 4)

    try:
        ac_vals = acf(signal, nlags=n_lags, fft=True)
        # Check for NaNs immediately in output
        if np.any(np.isnan(ac_vals)):
            return [0.0] * (n_lags + 1 + 4)
    except Exception:
        return [0.0] * (n_lags + 1 + 4)

    features = list(ac_vals)

    # 2. Largest Local Max
    val_max, idx_max = get_largest_local_max(ac_vals[1:], order=10)
    lag_hz_max = fs / (idx_max + 1) if (idx_max + 1) > 0 else 0

    # 3. Largest Local Min
    val_min_neg, idx_min = get_largest_local_max(-ac_vals[1:], order=10)
    val_min = -val_min_neg
    lag_hz_min = fs / (idx_min + 1) if (idx_min + 1) > 0 else 0

    features.extend([val_max, lag_hz_max, val_min, lag_hz_min])

    # Final sanity check for NaNs in the scalar features
    return [0.0 if np.isnan(x) else x for x in features]


def get_autocorr_features(X, n_lags=60, fs=2000):
    n_samples = X.shape[0]

    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_norm = np.divide(
        X - X_mean, X_std, out=np.zeros_like(X), where=X_std > 1e-10
    )

    all_features = []
    for i in range(n_samples):
        row_feats = _get_single_autocorr_features(X_norm[i], n_lags, fs)
        all_features.append(row_feats)

    return np.array(all_features)


def get_features(X, fs=2000, n_fourier_bins=50, n_lags=60, **kwargs):
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X = np.nan_to_num(X)

    dist_feats = get_distribution_features(X)
    fourier_feats = get_fourier_features(X, fs=fs, n_bins=n_fourier_bins)
    autocorr_feats = get_autocorr_features(X, n_lags=n_lags, fs=fs)

    X_feats = np.hstack([dist_feats, fourier_feats, autocorr_feats])

    return np.nan_to_num(X_feats)
