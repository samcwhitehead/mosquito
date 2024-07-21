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


# ------------------------------------------------------------------------------
# Set plotting parameters
def set_plot_params(plot_type='paper'):
    from matplotlib import rcParams
    # generate a dictionary called fig params that encodes information we care about
    if plot_type == 'powerpoint':
        fontsize = 14
        figsize = (10, 7.5)
        subplot_left = 0.15
        subplot_right = 0.85
        subplot_top = 0.8
        subplot_bottom = 0.15
        axis_linewidth = 1

    if plot_type == 'poster':
        fontsize = 18
        figsize = (10, 7.5)
        subplot_left = 0.15
        subplot_right = 0.85
        subplot_top = 0.8
        subplot_bottom = 0.15
        axis_linewidth = 1.5

    elif plot_type == 'paper':
        fontsize = 8
        figsize = (8, 8)
        subplot_left = 0.2
        subplot_rvight = 0.8
        subplot_top = 0.8
        subplot_bottom = 0.2
        axis_linewidth = 1

    fig_params = {
          'font.family': 'sans-serif',
          'font.serif': 'Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman',
          'font.sans-serif': 'arial, Helvetica, Avant Garde, Computer Modern Sans serif',
          'font.cursive': 'Zapf Chancery',
          'font.monospace': 'Courier, Computer Modern Typewriter',
          'font.size': fontsize,
          'axes.labelsize': fontsize,
          'axes.linewidth': axis_linewidth,
          'xtick.major.width': axis_linewidth,
          'xtick.minor.width': axis_linewidth,
          'ytick.major.width': axis_linewidth,
          'ytick.minor.width': axis_linewidth,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'figure.figsize': figsize,
    }
    rcParams.update(fig_params)

    return fig_params


# ------------------------------------------------------------------------------
# make axes conform to plotting style
def my_adjust_axes(ax, keep_spines=['left', 'bottom'], xticks=None, yticks=None,
                   offset_length=10, tick_length=5, trim_ticks_flag=True):
    """
    Convenience function to adjust axes to match plotting style

    Borrows heavily from flyplotlib:
    https://github.com/florisvb/FlyPlotLib/

    Args:
        ax: matplotlib axis object
        keep_spines: list of spines to keep
        xticks: array containing x tick mark locations
        yticks: array containing y tick mark locations
        offset_length: amount to offset axis rulers by
        tick_length: length to set axis ticks to
        trim_ticks_flag: bool, should we try to reduce tick count?

    Returns:
        ax: matplotlib axis object

    TODO:
        - allow different offsets for each spine
    """
    # make sure keep_spines is a list
    if type(keep_spines) is not list:
        keep_spines = [keep_spines]

    # check if we should remove all axes and just return
    if 'none' in keep_spines:
        for loc, spine in ax.spines.items():
            spine.set_color('none')  # don't draw spine
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(length=0)
        return ax

    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    elif not isinstance(xticks, np.ndarray):
        xticks = np.asarray(xticks)

    if yticks is None:
        yticks = ax.get_yticks()
    elif not isinstance(yticks, np.ndarray):
        yticks = np.asarray(yticks)

    # sometimes we get ticks outside of limits that confuse the issue--remove
    xlim = ax.get_xlim()
    xticks = xticks[(xticks >= xlim[0]) & (xticks <= xlim[1])]
    ylim = ax.get_ylim()
    yticks = yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]

    # get spines from axis object
    spines = ax.spines
    spine_locs = [loc for loc in spines.keys()]

    # remove spines that we don't want
    rm_spines = [loc for loc in spine_locs if loc not in keep_spines]
    for loc in rm_spines:
        ax.spines[loc].set_visible(False)

    # turn off ticks where there is no spine
    if 'left' in keep_spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in keep_spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in keep_spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in keep_spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    # for spines we want to keep, offset them
    for loc in keep_spines:
        # move spines outward
        ax.spines[loc].set_position(('outward', offset_length))

        # adjust spine size based on tick lengths
        if loc in ['left', 'right']:
            ticks = yticks
        if loc in ['top', 'bottom']:
            ticks = xticks
        if ticks is not None and len(ticks) > 0:
            ax.spines[loc].set_bounds(ticks[0], ticks[-1])

        # trim ticks down to minimum?
        if trim_ticks_flag:
            # take only ticks at limits or ticks at limits + middle
            if (ticks.size > 3) & (ticks.size % 2 == 1):
                ticks = ticks[[0, int(np.floor(xticks.size / 2)), -1]]
            elif (ticks.size > 3) & (ticks.size % 2 == 0):
                ticks = ticks[[0, -1]]

            # update tick values in axis
            if loc in ['left', 'right']:
                ax.yaxis.set_ticks(ticks)
            if loc in ['top', 'bottom']:
                ax.xaxis.set_ticks(ticks)

    # also just set ticks inward
    ax.tick_params(direction='in', length=tick_length)

    # return
    return ax


# -----------------------------------------------------------------------
# add scalebar to axis
def my_add_scalebar(ax, scalebar_bounds, linewidth=1.5, spine='bottom', units='',
                    ticklength=0, offset_length=10, label_mult_factor=1):
    """
    Function to convert an axis spine into a scalebar

    Args:
        ax: matplotlib axis object
        scalebar_bounds: 2 element tuple giving start and end of scalebar
        linewidth: width of scalebar
        spine: which spine to turn into scalebar
        units: string giving units of scalebar
        ticklength: length of ticks to use for scalebar
        offset_length: how much to move scalebar outward by
        label_mult_factor: number to multiply by scalebar size to get
            value for label. allows, e.g. to use ms in a plot where the
            data is in s

    Returns:
        ax
    """
    # set spine to be visible
    ax.spines[spine].set_visible(True)
    # reduce its length
    ax.spines[spine].set_bounds([scalebar_bounds[0], scalebar_bounds[1]])
    # set thickness
    ax.spines[spine].set_linewidth(linewidth)
    # move outward
    ax.spines[spine].set_position(('outward', offset_length))

    # label scalebar
    scalebar_center = (scalebar_bounds[0] + scalebar_bounds[1]) / 2
    scalebar_size = scalebar_bounds[1] - scalebar_bounds[0]
    label_val = label_mult_factor*scalebar_size
    if label_val == round(label_val):
        label_val = round(label_val)

    if spine in ['bottom', 'top']:
        ax.set_xticks([scalebar_center])
        ax.set_xticklabels([f'{label_val} {units}'])
        ax.tick_params(axis='x', length=ticklength)
    elif spine in ['left', 'right']:
        ax.set_yticks([scalebar_center])
        ax.set_yticklabels([f'{label_val} {units}'])
        ax.tick_params(axis='y', length=ticklength)

    # return
    return ax

