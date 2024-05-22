"""
Helper functions for analyzing mosquito data

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import numpy as np
from scipy import signal


# ---------------------------------------
#  FUNCTIONS
# ---------------------------------------

# ---------------------------------------------------------------------------------
def idx_by_thresh(sig, thresh=0.1):
    """
    Returns a list of index lists, where each index indicates sig > thresh

    If I'm remembering correctly, there's some ambiguity in the edge cases
    """
    # import numpy as np
    idxs = np.squeeze(np.argwhere((sig > thresh).astype(int)))
    try:
        split_idxs = np.squeeze(np.argwhere(np.diff(idxs) > 1))
    except IndexError:
        # print 'IndexError'
        return None
    # split_idxs = [split_idxs]
    if split_idxs.ndim == 0:
        split_idxs = np.array([split_idxs])
    # print split_idxs
    try:
        idx_list = np.split(idxs, split_idxs)
    except ValueError:
        # print 'value error'
        np.split(idxs, split_idxs)
        return None
    idx_list = [x[1:] for x in idx_list]
    idx_list = [x for x in idx_list if len(x) > 0]
    return idx_list


# ---------------------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, sampling_period, order=5, btype='band'):
    """
    make second-order sections representation of butter bandpass filter
    """

    sampling_frequency = 1.0 / sampling_period
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype=btype, output='sos')
    return sos


# ---------------------------------------------------------------------------------
def butter_bandpass_filter(data, lowcut, highcut, sampling_period, order=5,
                           btype='band'):
    """
    filter a time series using a butter bandpass filter
    """
    sos = butter_bandpass(lowcut, highcut, sampling_period, order=order,
                          btype=btype)
    y = signal.sosfiltfilt(sos, data)
    return y


# ------------------------------------------------------------------------------
def butter_lowpass(lowcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0 / sampling_period
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    sos = signal.butter(order, low, btype='low', output='sos')
    return sos


# ------------------------------------------------------------------------------
def butter_lowpass_filter(data, lowcut, sampling_period, order=5):
    import scipy.signal
    sos = butter_lowpass(lowcut, sampling_period, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


# ------------------------------------------------------------------------------
def butter_highpass(highcut, sampling_period, order=5):
    import scipy.signal
    sampling_frequency = 1.0 / sampling_period
    nyq = 0.5 * sampling_frequency
    high = highcut / nyq
    sos = signal.butter(order, high, btype='high', output='sos')
    return sos


# ------------------------------------------------------------------------------
def butter_highpass_filter(data, highcut, sampling_period, order=5):
    import scipy.signal
    sos = butter_highpass(highcut, sampling_period, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


# ---------------------------------------------------------------------------------
def iir_notch(cut_freq, sampling_period, Q=30):
    """
    generate filter coefficients for iir notch
    """
    sampling_frequency = 1.0 / sampling_period
    b, a = signal.iirnotch(cut_freq, Q, fs=sampling_frequency)
    return b, a


# ---------------------------------------------------------------------------------
def iir_notch_filter(data, cut_freq, sampling_period, Q=30):
    """
    filter data according to iir notch
    """
    b, a = iir_notch(cut_freq, sampling_period, Q=Q)
    y = signal.filtfilt(b, a, data)
    return y


# ------------------------------------------------------------------------------
# rolling window that avoids looping
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# ------------------------------------------------------------------------------
# hampel filter
def hampel(x, k=7, t0=3):
    """taken from stack overflow
    x= 1-d numpy array of numbers to be filtered
    k= number of items in window/2 (# forward and backward wanted to capture in median filter)
    t0= number of standard deviations to use; 3 is default
    """
    dk = int((k - 1) / 2)
    y = x.copy()  # y is the corrected series
    L = 1.4826

    # calculate rolling median
    rolling_median = np.nanmedian(rolling_window(y, k), -1)
    rolling_median = np.concatenate((y[:dk], rolling_median, y[-dk:]))

    # compare rolling median to value at each point
    difference = np.abs(rolling_median - y)
    median_abs_deviation = np.nanmedian(rolling_window(difference, k), -1)
    median_abs_deviation = np.concatenate((difference[:dk], median_abs_deviation,
                                           difference[-dk:]))

    # determine where data exceeds t0 standard deviations from the local median
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold

    y[outlier_idx] = rolling_median[outlier_idx]

    return y, outlier_idx


# ------------------------------------------------------------------------------
# rolling average filter
def moving_avg(x, k=3):
    """
    taken from stack overflow
    x= 1-d numpy array of numbers to be filtered
    k= number of items in window/2 (# forward and backward wanted to capture in filter)
    """
    dk = int((k - 1) / 2)
    y = x.copy()  # y is the corrected series

    # calculate rolling median
    rolling_mean = np.nanmean(rolling_window(y, k), -1)
    rolling_mean = np.concatenate((y[:dk], rolling_mean, y[-dk:]))

    return rolling_mean