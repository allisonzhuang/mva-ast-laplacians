from sklearn.model_selection import train_test_split
from itertools import tee

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

import numpy as np
import pandas as pd

from aeon.datasets import load_classification
from dtw import dtw
import time
from scipy.cluster import hierarchy
from scipy.stats import f_oneway


from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector

from alphacsc import learn_d_z
try:
    from alphacsc.utils import construct_X
except:
    from alphacsc.utils.convolution import construct_X

from scipy.signal import argrelmax


def _parse_stocks(data, period, pred_days):
    N = data.size
    series = []
    outcomes = []

    for i in range(N - period - pred_days):
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
        X, y = _parse_stocks(data, kwargs.get("period", 60), kwargs.get("pred_days", 6))
    elif dataset_name == "amazon":
        data = np.genfromtxt("AMZN.csv", delimiter=",", skip_header=1, usecols=[1])
        X, y = _parse_stocks(data, kwargs.get("period", 60), kwargs.get("pred_days", 6))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")

    if split_data:
        return train_test_split(X, y, test_size=0.2)
    return X, y

def protected_log2(p):
  #debug this bc i'm pretty sure stuff still isn't working here.
  return np.where(p > 0, np.log2(p), 0) #replace w -inf?

def get_rolling_features(df, window_sizes):
  #rolling mean, std, autocorrelation with lag of 5 days, std of daily return.
  for window_size in window_sizes:
      #figure out how to make this fast
      window_str = str(window_size)

      df[f'r_mean_{window_str}'] = df['Adj Close'].rolling(window=window_size).mean()
      df[f'r_std_{window_str}'] = df['Adj Close'].rolling(window=window_size).std()

      df[f'lag5_corr_{window_str}'] = df['Adj Close'].rolling(window=window_size).corr(df['Adj Close'].shift(5))
      df[f'return_std_{window_str}'] = df['Daily_Return'].rolling(window=window_size).std()

def extract_and_save_features(ticker, window_sizes=[20,60], A_size=4):
    #ticker: AMZN, MSFT;
    #window_sizes: list containing the number of days for each window
    #A_size is number of symbols in dictionary

    input_filename = f'{ticker}.csv'
    output_filename = f'{ticker}_features.csv'

    #convert dates to datetime format, calculate daily return
    #hopefully adj close means what i think it does lol
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Daily_Return'] = df['Adj Close'].pct_change()

    get_rolling_features(df, window_sizes) #4 features per window size
    # get_symbolic_features(df, window_sizes, A=A_size)

    feature_cols = [col for col in df.columns if col not in ['Adj Close', 'Daily_Return']]
    df_features = df[feature_cols].dropna()

    print(f"features extracted: {df_features.shape[1]}")
    print(df_features.columns.tolist())

    df_features.to_csv(output_filename)

def get_dtw_distance(signal_1,signal_2):
    alignment = dtw(signal_1, signal_2, keep_internals=True)
    return alignment.distance

def get_euclidean_distance(signal_1,signal_2):
    return np.linalg.norm(signal_1 - signal_2)

def display_distance_matrix_as_table(
    distance_matrix, labels=None, figsize=(8, 2)
):
#only using this one afaik
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")
    norm = mpl.colors.Normalize()
    cell_colours_hex = np.empty(shape=distance_matrix.shape, dtype=object)
    cell_colours_rgba = plt.get_cmap("magma")(norm(distance_matrix))

    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[0]):
            cell_colours_hex[i, j] = rgb2hex(
                cell_colours_rgba[i, j], keep_alpha=True
            )
            cell_colours_hex[j, i] = cell_colours_hex[i, j]

    if labels is not None:
        _ = ax.table(
            cellText=distance_matrix,
            colLabels=labels,
            rowLabels=labels,
            loc="center",
            cellColours=cell_colours_hex,
        )
    else:
        _ = ax.table(
            cellText=distance_matrix,
            loc="center",
            cellColours=cell_colours_hex,
        )

    return ax

def plot_CDL(signal, Z, D, figsize=(15, 10)):
    """Plot the learned dictionary `D` and the associated sparse codes `Z`.

    `signal` is an univariate signal of shape (n_samples,) or (n_samples, 1).
    """
    (atom_length, n_atoms) = np.shape(D)
    plt.figure(figsize=figsize)
    plt.subplot(n_atoms + 1, 3, (2, 3))
    plt.plot(signal)
    for i in range(n_atoms):
        plt.subplot(n_atoms + 1, 3, 3 * i + 4)
        plt.plot(D[:, i])
        plt.subplot(n_atoms + 1, 3, (3 * i + 5, 3 * i + 6))
        plt.plot(Z[:, i])
        plt.ylim((np.min(Z), np.max(Z)))

def get_n_largest(
    arr: np.ndarray, n_largest: int = 3) -> (np.ndarray, np.ndarray):
    """Return the n largest values and associated indexes of an array.

    (In decreasing order of value.)
    """
    indexes = np.argsort(arr)[-n_largest:][::-1]
    if n_largest == 1:
        indexes = np.array(indexes)
    values = np.take(arr, indexes)
    return values, indexes

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_largest_local_max(signal1D: np.ndarray, order: int = 1):
    """Return the largest local max and the associated index in a tuple.

    This function uses `order` points on each side to use for the comparison.
    """
    all_local_max_indexes = argrelmax(signal1D, order=order)[0]
    all_local_max = np.take(signal1D, all_local_max_indexes)
    largest_local_max_index = all_local_max_indexes[all_local_max.argsort()[-1]]

    return signal1D[largest_local_max_index], largest_local_max_index

from scipy.stats import skew, kurtosis

def get_distribution_features(signal: np.ndarray) -> dict:
    res_dict = dict()
    res_dict["mean"] = signal.mean()
    res_dict["std"] = signal.std()
    res_dict["min"] = signal.min()
    res_dict["max"] = signal.max()
    res_dict["skew"] = skew(signal)
    res_dict["kurtosis"] = kurtosis(signal)
    res_dict["25%"] = np.percentile(signal, 25)
    res_dict["50%"] = np.percentile(signal, 50)
    res_dict["75%"] = np.percentile(signal, 75)
    return res_dict

from scipy.fft import rfft, rfftfreq

def get_fourier_features(signal: np.ndarray, n_bins: int = 100) -> dict:
    """The signal is assumed to be centered and scaled to unit variance."""
    n_samples = signal.shape[0]
    fourier = abs(rfft(signal))
    freqs = rfftfreq(n=n_samples, d=1.0 / FREQUENCY)
    res_dict = dict()

    freq_bins = np.linspace(0, FREQUENCY / 2, n_bins + 1)
    for f_min, f_max in pairwise(freq_bins):
        keep = (f_min <= freqs) & (freqs < f_max)
        res_dict[f"fourier_{f_min:.0f}-{f_max:.0f}_Hz"] = np.log(
            np.sum(fourier[keep] ** 2)
        )
    return res_dict

from statsmodels.tsa.stattools import acf

def samples_to_hz(lag_in_samples, FREQUENCY):
    return FREQUENCY / lag_in_samples

def get_autocorr_features(signal: np.ndarray, n_lags: int = 200) -> dict:
    auto_corr = acf(signal, nlags=n_lags, fft=True)
    res_dict = dict()
    for lag, auto_corr_value in enumerate(auto_corr):
        res_dict[f"autocorrelation_{lag}_lag"] = auto_corr_value

    local_max, local_argmax = get_largest_local_max(auto_corr[1:], order=10)
    local_argmax += 1  # to account for the lag=0 removed before
    local_min, local_argmin = get_largest_local_max(-auto_corr[1:], order=10)
    local_min = -local_min
    local_argmin += 1  # to account for the lag=0 removed before
    res_dict["largest_local_max_autocorrelation"] = local_max
    res_dict["lag_largest_local_max_autocorrelation_Hz"] = samples_to_hz(local_argmax, FREQUENCY)
    res_dict["largest_local_min_autocorrelation"] = local_min
    res_dict["lag_largest_local_min_autocorrelation_Hz"] = samples_to_hz(local_argmin, FREQUENCY)
    return res_dict

def get_features(signal: np.ndarray) -> dict:
    res_dict = dict()

    # stats
    res_dict.update(get_distribution_features(signal))

    # spectral
    signal -= signal.mean()
    signal /= signal.std()
    res_dict.update(get_fourier_features(signal, n_bins=50))

    # autocorrelation
    res_dict.update(get_autocorr_features(signal, n_lags=200))

    return res_dict

def samples_to_hz(lag_in_samples, FREQUENCY):
    # This function is not strictly needed for the stock data which doesn't have a defined Hz,
    # but we'll keep it for structural completeness.
    return FREQUENCY / lag_in_samples if lag_in_samples > 0 else 0

def get_largest_local_max(signal1D: np.ndarray, order: int = 1):
    """
    Return the largest local max and the associated index in a tuple.
    Safely handles cases where no local maxima are found.
    """
    all_local_max_indexes = argrelmax(signal1D, order=order)[0]

    if all_local_max_indexes.size == 0:
        return np.nan, np.nan

    all_local_max = np.take(signal1D, all_local_max_indexes)
    largest_local_max_index = all_local_max_indexes[np.argmax(all_local_max)]

    return signal1D[largest_local_max_index], largest_local_max_index

def get_distribution_features(signal: np.ndarray) -> dict:
    """Extract standard distribution features."""
    res_dict = dict()

    if signal.size == 0:
        return {f: np.nan for f in ["mean", "std", "min", "max", "skew", "kurtosis", "25%", "50%", "75%"]}

    res_dict["mean"] = signal.mean()
    res_dict["std"] = signal.std()
    res_dict["min"] = signal.min()
    res_dict["max"] = signal.max()
    res_dict["skew"] = skew(signal)
    res_dict["kurtosis"] = kurtosis(signal)
    res_dict["25%"] = np.percentile(signal, 25)
    res_dict["50%"] = np.percentile(signal, 50)
    res_dict["75%"] = np.percentile(signal, 75)
    return res_dict

def get_fourier_features(signal: np.ndarray, n_bins: int = 100, FREQUENCY: int = 2000) -> dict:
    """
    Extract spectral features by binning the power spectrum of the FFT.
    The signal is assumed to be centered and scaled to unit variance.
    """
    n_samples = signal.shape[0]
    if n_samples == 0:
         return {f"fourier_{i}-{i+1}_Hz": np.nan for i in range(n_bins)}

    fourier = abs(rfft(signal))
    freqs = rfftfreq(n=n_samples, d=1.0 / FREQUENCY)
    res_dict = dict()

    freq_bins = np.linspace(0, FREQUENCY / 2, n_bins + 1)

    def safe_log(arr):
        return np.log(np.where(arr > 0, arr, 1e-10))

    for f_min, f_max in pairwise(freq_bins):
        keep = (f_min <= freqs) & (freqs < f_max)

        power = np.sum(fourier[keep] ** 2)
        res_dict[f"fourier_{f_min:.0f}-{f_max:.0f}_Hz"] = safe_log(power)

    return res_dict

def get_autocorr_features(signal: np.ndarray, n_lags: int = 200, order: int = 10, FREQUENCY: int = 2000) -> dict:
    """
    Extract features from the autocorrelation function.
    Safely handles finding local max/min by returning NaN if none are found.
    """
    res_dict = dict()
    n_samples = signal.shape[0]

    max_lag = min(n_samples - 1, n_lags)
    if max_lag <= 0:
        return res_dict

    auto_corr = acf(signal, nlags=max_lag, fft=True)

    for lag, auto_corr_value in enumerate(auto_corr):
        res_dict[f"autocorrelation_{lag}_lag"] = auto_corr_value

    local_max, local_argmax_index = get_largest_local_max(auto_corr[1:], order=order)

    if np.isnan(local_max):
        # no local max found
        res_dict["largest_local_max_autocorrelation"] = 0
        res_dict["lag_largest_local_max_autocorrelation_Hz"] = np.nan
    else:
        local_argmax = local_argmax_index + 1
        res_dict["largest_local_max_autocorrelation"] = local_max
        res_dict["lag_largest_local_max_autocorrelation_Hz"] = samples_to_hz(local_argmax, FREQUENCY)

    local_min_neg, local_argmin_index = get_largest_local_max(-auto_corr[1:], order=order)

    if np.isnan(local_min_neg):
        # no local minimum found
        res_dict["largest_local_min_autocorrelation"] = 0
        res_dict["lag_largest_local_min_autocorrelation_Hz"] = np.nan
    else:
        # local_argmin_index is relative to auto_corr[1:], so add 1 for true lag
        local_argmin = local_argmin_index + 1
        res_dict["largest_local_min_autocorrelation"] = -local_min_neg
        res_dict["lag_largest_local_min_autocorrelation_Hz"] = samples_to_hz(local_argmin, FREQUENCY)

    return res_dict

def get_features(signal: np.ndarray, FREQUENCY: int = 2000, N_FOURIER_BINS: int = 50, N_LAGS: int = 60) -> dict:
    """
    Extract a comprehensive set of features from a single time series signal.
    """
    res_dict = dict()

    res_dict.update(get_distribution_features(signal))

    scaled_signal = signal.copy()

    if scaled_signal.std() > 1e-10:
        scaled_signal -= scaled_signal.mean()
        scaled_signal /= scaled_signal.std()
    else:
        fourier_features = {f"fourier_{f_min:.0f}-{f_max:.0f}_Hz": 0 for f_min, f_max in pairwise(np.linspace(0, FREQUENCY / 2, N_FOURIER_BINS + 1))}
        res_dict.update(fourier_features)

        autocorr_features = {f"autocorrelation_{lag}_lag": 0 for lag in range(N_LAGS + 1)}
        autocorr_features["largest_local_max_autocorrelation"] = 0
        autocorr_features["lag_largest_local_max_autocorrelation_Hz"] = np.nan
        autocorr_features["largest_local_min_autocorrelation"] = 0
        autocorr_features["lag_largest_local_min_autocorrelation_Hz"] = np.nan
        res_dict.update(autocorr_features)

        return res_dict


    res_dict.update(get_fourier_features(scaled_signal, n_bins=N_FOURIER_BINS, FREQUENCY=FREQUENCY))

    res_dict.update(get_autocorr_features(scaled_signal, n_lags=N_LAGS, order=10, FREQUENCY=FREQUENCY))

    return res_dict

def get_rolling_features_refactored(X, window_size):
  # window_size = X.shape[1]
  res_list = []

  for signal in X:
    res_dict = {}
    res_dict['r_mean'] = np.mean(X)
    res_dict['r_std'] = np.std(X)
    res_dict['return_std'] = np.std(np.divide(X[0:-1], X[1:]))

    res_list.append(res_dict)
  return res_list

def find_low_variance_features(all_features, variance_threshold=0.1):
    #dropping low variance features:

    low_variance_features = all_features.std() < variance_threshold
    low_variance_features = low_variance_features[
        low_variance_features
    ].index.to_numpy()
    print(f"There are {len(low_variance_features)} features to drop.")
    print(low_variance_features)

    return low_variance_features

def drop_low_variance_features(all_features, variance_threshold=0.1):
    features_to_drop = find_low_variance_features(all_features, variance_threshold=variance_threshold)
    all_features.drop(columns=features_to_drop, inplace=True, errors="ignore")
    print(f"There are {all_features.shape[1]} features left.")