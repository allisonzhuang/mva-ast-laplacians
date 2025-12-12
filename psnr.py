import numpy as np

def psnr_ecg(denoised_signal, original_signal, max_amplitude=None):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) for 1D signals like ECG.

    Args:
        
        denoised_signal (np.ndarray): The processed (denoised) signal = I
        original_signal (np.ndarray): The clean (ground truth) signal = K
        max_amplitude (float): The maximum possible amplitude of the signal.
                               If None, it uses the maximum value found in the original_signal.
                               For clinical ECG, this might be a fixed mV value.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    original = np.asarray(original_signal)
    denoised = np.asarray(denoised_signal)
    
    if original.shape != denoised.shape:
        raise ValueError("Original and denoised signals must have the same shape.")

    mse = np.mean((original - denoised) ** 2)

    if mse == 0:
        return float('inf') 

    if max_amplitude is None:
        R = np.max(np.abs(original))
    else:
        R = max_amplitude

    psnr_value = 10 * np.log10(R**2 / mse)
    num = R**2
    denom = mse
    print(num, denom)
    return psnr_value

# Example Usage:
# Assume you have your clean_ecg and denoised_ecg numpy arrays
# psnr_score = psnr_ecg(clean_ecg, denoised_ecg, max_amplitude=1.0)