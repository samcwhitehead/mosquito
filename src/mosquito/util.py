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
    just use pandas
    x= 1-d numpy array of numbers to be filtered
    k= number of items in window/2 (# forward and backward wanted to capture in filter)
    """
    # dk = int((k - 1) / 2)
    # y = x.copy()  # y is the corrected series
    #
    # # calculate rolling median
    # rolling_mean = np.nanmean(rolling_window(y, k), -1)
    # rolling_mean = np.concatenate((y[:dk], rolling_mean, y[-dk:]))
    # return rolling_mean
    import pandas as pd
    x_ser = pd.Series(data=x)
    sliding_average = x_ser.rolling(k).mean()
    return sliding_average.values


# ------------------------------------------------------------------------------
# Estimate local slope for a sequence of points, using a sliding window
def moving_slope(vec, supportlength=3, modelorder=1, dt=1):
    """
    Estimate local slope for a sequence of points, using a sliding window

    movingslope uses filter to determine the slope of a curve stored
    as an equally (unit) spaced sequence of points. A patch is applied
    at each end where filter will have problems. A non-unit spacing
    can be supplied.

    Note that with a 3 point window and equally spaced data sequence,
    this code should be similar to gradient. However, with wider
    windows this tool will be more robust to noisy data sequences.

    From https://www.mathworks.com/matlabcentral/fileexchange/16997-movingslope

    Arguments:
        vec - row of column vector, to be differentiated. vec must be of
            length at least 2.

        supportlength - (OPTIONAL) scalar integer - defines the number of
            points used for the moving window. supportlength may be no
            more than the length of vec.

            supportlength must be at least 2, but no more than length(vec)

            If supportlength is an odd number, then the sliding window
            will be central. If it is an even number, then the window
            will be slid backwards by one element. Thus a 2 point window
            will result in a backwards differences used, except at the
            very first point, where a forward difference will be used.

            DEFAULT: supportlength = 3

        modelorder - (OPTIONAL) - scalar - Defines the order of the windowed
            model used to estimate the slope. When model order is 1, the
            model is a linear one. If modelorder is less than supportlength-1.
            then the sliding window will be a regression one. If modelorder
            is equal to supportlength-1, then the window will result in a
            sliding Lagrange interpolant.

            modelorder must be at least 1, but not exceeding
            min(10,supportlength-1)

            DEFAULT: modelorder = 1

        dt - (OPTIONAL) - scalar - spacing for sequences which do not have
            a unit spacing.

            DEFAULT: dt = 1

    Returns:
        Dvec = vector of derivative estimates, Dvec will be of the same size
            and shape as is vec.
    """

    # helper function to get filter coefficients
    def getcoef(t, supportlength, modelorder):
        a = np.tile(t, (modelorder + 1, 1)).T ** np.tile(np.arange(modelorder + 1), (supportlength, 1))
        pinva = np.linalg.pinv(a)
        coef = pinva[1, :]
        return coef

    # length of input vector
    n = vec.size

    # build filter coefficients to estimate slope
    if (supportlength % 2) == 1:
        parity = 1  # odd parity
    else:
        parity = 0  # even parity

    s = (supportlength - parity) / 2
    t = np.arange(-s + 1 - parity, s + 1)
    coef = getcoef(t, supportlength, modelorder)

    # Apply the filter to the entire vector
    f = signal.lfilter(-coef, 1, vec)
    Dvec = np.zeros(vec.shape)
    idx = slice(int(s + 1), int(s + n - supportlength))
    Dvec[idx] = f[supportlength:-1]

    # Patch each end
    for ith in range(int(s)):
        # patch the first few points
        t = np.arange(supportlength) - ith
        coef = getcoef(t, supportlength, modelorder)
        Dvec[ith] = np.dot(coef, vec[:supportlength])

        # patch the end points
        if ith < (s + parity):
            t = np.arange(supportlength) - supportlength + ith - 1
            coef = getcoef(t, supportlength, modelorder)
            Dvec[n - ith - 1] = np.dot(coef, vec[n + np.arange(supportlength) - supportlength])

    return Dvec