"""
Code to analyze EMG bursts (e.g. to detect the number of spikes per burst)

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

try:
    from util import butter_bandpass_filter, moving_avg, moving_slope
    from process_abf import load_processed_data, detrend_emg
except ModuleNotFoundError:
    from .util import butter_bandpass_filter, moving_avg, moving_slope
    from .process_abf import load_processed_data, detrend_emg
# ---------------------------------------
# PARAMS
# ---------------------------------------
BURST_WINDOW = 1024  # was 1028? # number of time points to get when collecting spike windows


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
# ---------------------------------------------------------------------------------
def filter_for_spike_detection(emg, fs, wbf, flying_idx=None,
                               filt_order=25, moving_avg_win=7):
    """
    Function to filter emg data so we can detect spikes within bursts of it
    Mainly using this to remove noise at wingbeat frequency, but also a little
    smoothing helps with peak detection

    Args:
        emg: numpy array containing emg measurements
        fs: sampling freq (Hz)
        wbf: np array containing wingbeat frequency
        flying_idx: index of flying bouts in wbf data
        filt_order: which order to use for butterworth filter
        moving_avg_win: window size for rolling mean filter

    Returns:
        emg_filt_sd: emg data filtered for spike detection

    """
    # remove wingbeat noise
    if flying_idx is None:
        flying_idx = np.ones_like(wbf, dtype=bool)

    lowcut = np.floor(np.min(wbf[flying_idx]))
    highcut = np.ceil(np.max(wbf[flying_idx]))
    emg_filt_sd = butter_bandpass_filter(emg, lowcut, highcut, 1 / fs,
                                         order=filt_order, btype='stop')

    # do moving avg filter
    emg_filt_sd = moving_avg(emg_filt_sd, k=moving_avg_win)

    return emg_filt_sd


# ---------------------------------------------------------------------------------
def detect_burst_peaks(burst_waveform, min_height_factor=0.3, min_prom_factor=0.025,
                       min_distance=40, burst_waveform_filt=None, detrend_flag=True,
                       viz_flag=False):
    """
    Function to detect the peaks (spikes) within a power muscle burst

    Args:
        burst_waveform: numpy array containing waveform of burst to detect spikes in
        min_height_factor: number to multiply by burst_waveform height
            to get minimum height in peak detection
        min_prom_factor: number to multiply by burst_waveform height
            to get minimum prominence in peak detection
        min_distance: minimum distance (x direction) between adjacent peaks
        burst_waveform_filt: filtered burst waveform data (smooth enough to not
            show bursting)
        detrend_flag: bool, do linear detrend of burst waveform
        viz_flag: bool, visualize peak detection?

    Returns:
        peaks: a list containing spike indices

    """
    # detrend burst waveform and subtract off initial value
    if detrend_flag:
        burst_waveform = signal.detrend(burst_waveform)  # detrend data
    burst_waveform -= burst_waveform[0]

    # set some parameters for peak detection
    min_height = min_height_factor * np.max(burst_waveform)
    min_prominence = min_prom_factor * np.max(burst_waveform)

    # do peak detection
    peaks, props = signal.find_peaks(burst_waveform,
                                     distance=min_distance,
                                     height=(min_height, None),
                                     prominence=(min_prominence, None))

    # use filtered data to check peak location
    if burst_waveform_filt is not None:

        # get maximum of filtered burst
        burst_max = np.max(burst_waveform_filt)
        max_idx = np.argmax(burst_waveform_filt)

        # first point to the left of the maximum that dips below X%
        pre_idx = np.where(burst_waveform_filt[:max_idx] > 0.2*burst_max)[0]
        if pre_idx.size > 0:
            left_idx = pre_idx[0]
        else:
            left_idx = 0

        # first point to the right of the maximum that dips below X%
        post_idx = np.where(burst_waveform_filt[max_idx:] < 0.2*burst_max)[0]
        if post_idx.size > 0:
            right_idx = post_idx[0]
            right_idx += max_idx
        else:
            right_idx = burst_waveform_filt.size - 1

        # use these to restrict peak locations
        keep_idx = (peaks > left_idx) & (peaks < right_idx)
        peaks = peaks[keep_idx]

    # visualize?
    if viz_flag:
        fig, ax = plt.subplots()
        tmp_range = np.arange(burst_waveform.size)
        ax.plot(tmp_range, burst_waveform)
        ax.plot(tmp_range[peaks], burst_waveform[peaks], 'rx')

        ax.set_xlabel('x (index)')
        ax.set_ylabel('emg (V)')
        ax.set_title(f'{peaks.size} peaks')
        plt.show()

    # check that we've found peaks
    if peaks.size < 1:
        print('Could not find peaks')
        return None

    return peaks


# ---------------------------------------------------------------------------------
def detect_burst_peaks_deriv(burst_waveform, deriv_height_factor=0.1,
                             min_height_factor=0.3, deriv_width_bounds=(20, 100),
                             burst_waveform_filt=None, refine_flag=True,
                             detrend_flag=True, viz_flag=False):
    """
    An alternate approach to detecting spikes within an EMG burst based on first
    finding peaks in the derivative of the signal. Works poorly on noisy data,
    but better on signals that have shallow peaks

    Args:
        burst_waveform: numpy array containing waveform of burst to detect spikes in
        deriv_height_factor: number to multiply by burst waveform derivative height
            to get minimum height in peak detection
        min_height_factor: number to multiply by burst_waveform height
            to get minimum height for peaks
        deriv_width_bounds: min and max *derivative* peak width, in index units
        burst_waveform_filt: filtered burst waveform data (smooth enough to not
            show bursting)
        refine_flag: try an additional refinement step for peak location at the end?
        detrend_flag: detrend burst waveform signal? Useful for high bg drift
        viz_flag: bool, visualize peak detection?

    Returns:
        peaks: a list containing spike indices

    TODO: make width bounds calculated rather than hard-coded

    """
    # detrend burst?
    if detrend_flag:
        burst_waveform = signal.detrend(burst_waveform)  # detrend data

    # find derivative points corresponding to pre-spike rises
    burst_waveform_dot = moving_slope(burst_waveform, supportlength=101, modelorder=7)
    height_bounds = (deriv_height_factor * np.max(burst_waveform_dot), None)
    deriv_peaks, props = signal.find_peaks(burst_waveform_dot,
                                           height=height_bounds,
                                           width=deriv_width_bounds)

    # check that we found peaks at all
    if (deriv_peaks.size < 2) or any(np.isnan(deriv_peaks)):
        return None

    # insert a guess for a right bound of the final peak
    final_deriv_peak = round(2 * np.mean(np.diff(deriv_peaks)) + deriv_peaks[-1])
    deriv_peaks = np.append(deriv_peaks, final_deriv_peak)

    # get the minima between derivative peaks (~ zero crossings)
    peaks = []
    for pk1, pk2 in zip(deriv_peaks[:-1], deriv_peaks[1:]):
        # get minimum between adjacent peaks
        peaks_tmp, _ = signal.find_peaks(burst_waveform[pk1:pk2])
        if peaks_tmp.size > 0:
            idx = peaks_tmp[0] + pk1
        else:
            idx = np.argmin(np.abs(burst_waveform_dot[pk1:pk2])) + pk1

        peaks.append(idx)

    peaks = np.asarray(peaks)

    # use filtered data to check peak location
    if burst_waveform_filt is not None:

        # get maximum of filtered burst
        burst_max = np.max(burst_waveform_filt)
        max_idx = np.argmax(burst_waveform_filt)

        # first point to the left of the maximum that dips below X%
        pre_idx = np.where(burst_waveform_filt[:max_idx] > 0.2 * burst_max)[0]
        if pre_idx.size > 0:
            left_idx = pre_idx[0]
        else:
            left_idx = 0

        # first point to the right of the maximum that dips below X%
        post_idx = np.where(burst_waveform_filt[max_idx:] < 0.2 * burst_max)[0]
        if post_idx.size > 0:
            right_idx = post_idx[0]
            right_idx += max_idx
        else:
            right_idx = burst_waveform.size - 1

        # use these to restrict peak locations
        keep_idx = (peaks > left_idx) & (peaks < right_idx)
        peaks = peaks[keep_idx]

    # take only peaks greater than the minimum peak height
    min_height = min_height_factor * np.max(burst_waveform)
    peaks = peaks[burst_waveform[peaks] > min_height]

    # attempt to refine peak estimation, now that we (hopefully) have
    # putative peaks that we're happy with
    if refine_flag:
        # create copy for refined output
        peaks_refined = np.zeros_like(peaks)

        # size of window to look near each peak
        refine_window_size = deriv_width_bounds[0]

        # loop over current peak estimates
        for ith, pk in enumerate(peaks):
            # look for maximum in small window around peak
            left_idx = np.max([pk - refine_window_size, 0])
            right_idx = np.min([pk + refine_window_size,
                               burst_waveform.size - 1])
            refine_window = np.arange(left_idx, right_idx)
            max_idx = np.argmax(burst_waveform[refine_window])
            peaks_refined[ith] = max_idx + refine_window[0]

        # replace old peaks with refined peaks
        peaks = peaks_refined

    # visualize?
    if viz_flag:
        fig, ax = plt.subplots()
        tmp_range = np.arange(burst_waveform.size)
        ax.plot(tmp_range, burst_waveform)
        ax.plot(tmp_range[peaks], burst_waveform[peaks], 'rx')

        ax.set_xlabel('x (index)')
        ax.set_ylabel('emg (V)')
        ax.set_title(f'{peaks.size} peaks')

        plt.show()

    # return
    return peaks


# ---------------------------------------------------------------------------------
def get_peak_props(peaks, emg, t, burst_idx):
    """
    Function to take an array of burst spike peaks and return properties
    that we might care about (peaks can be the output of detect_burst_peaks)

    Args:
        peaks: array containing estimated peak location for burst spikes
        emg: numpy array containing emg signal
        t: numpy array containing time measurements (same size as 'emg')
        burst_idx: index associated with detected burst
    Returns:
        peaks_df: dataframe containing info on peaks

    """
    # get height from left minima to peak height
    left_heights = list()
    left_heights.append(emg[peaks[0]])

    for ith in np.arange(1, peaks.size):
        left_min = np.min(emg[peaks[ith - 1]:peaks[ith]])
        left_height = emg[peaks[ith]] - left_min
        left_heights.append(left_height)

    # create peaks DataFrame
    peaks_dict = dict()
    peaks_dict['burst_idx'] = burst_idx * np.ones(peaks.shape, dtype=int)
    peaks_dict['spike_idx'] = peaks
    peaks_dict['spike_t'] = t[peaks]
    peaks_dict['spike_dt'] = t[peaks] - t[peaks[0]]  # time from first peak
    peaks_dict['spike_num'] = np.arange(peaks.size)
    peaks_dict['spike_height'] = emg[peaks]
    peaks_dict['spike_left_height'] = np.asarray(left_heights)
    peaks_df = pd.DataFrame.from_dict(peaks_dict)

    return peaks_df


# ---------------------------------------------------------------------------------
def realign_spikes(spikes, spike_idx, emg, window=BURST_WINDOW, thresh_factor=0.5,
                   viz_flag=False):
    """
    Helper function to align spike waveform traces based on when they
    first cross a given threshold

    Args:
        spikes: list containing isolated spike waveform arrays (data['spikes'])
        spike_idx: list of indices of detected spikes (data['spike_idx'])
        emg: array containing raw emg data (data['emg'])
        window: int giving the size of spike window to consider
        thresh_factor: number to multiply by mean spike height to get
            threshold whose crossing we align to
        viz_flag: boolean, should we visualize aligned spikes?

    Returns:
        spikes_realigned:  list containing isolated REALIGNED spike waveforms
        spike_realigned_idx: list of indices of REALIGNED spikes

    """
    # calculate thresh based on spike height and thresh_factor
    mean_spike_height = np.mean(np.max(np.vstack(spikes), axis=1))
    thresh = thresh_factor * mean_spike_height  # align waveforms to when spikes cross thresh

    # loop through and get aligned spike arrays
    spikes_realigned = list()
    spike_realigned_idx = list()

    for idx in spike_idx:
        # get spike in current window
        idx_range = np.arange(idx - window, idx + window)
        spike = emg[idx_range]

        # subtract off value at initial time point
        spike -= spike[0]

        # find first instance of thresh crossing
        try:
            thresh_idx = np.where(spike > thresh)[0][0]
        except IndexError:
            thresh_idx = window

        thresh_idx += (idx - window)
        spike_realigned_idx.append(thresh_idx)

        # get spike in this new window
        spike_new = emg[slice(thresh_idx - window, thresh_idx + 2 * window)]
        spikes_realigned.append(spike_new)

    # visualize?
    if viz_flag:
        # make figure and axis
        fig, ax = plt.subplots()

        # plot data
        for spike in spikes_realigned:
            ax.plot(spike, color='b', lw=0.1, alpha=0.1)

        # label axes
        ax.set_ylabel('emg (V)')
        ax.set_xlabel('index')

    return spikes_realigned, spike_realigned_idx


# ---------------------------------------------------------------------------------
def peak_detection_test(data, n_cols=3, filt_flag=True):
    """
    Function to test how well our peak detection is doing by visualizing the results

    Args:
        data: data dictionary from process_abf.py
        n_cols: number of subplot columns
        filt_flag: try an additional filtering of data to remove wbf noise?

    Returns:
        fig, ax_list: figure and axis handles for results

    """
    # -------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------
    # read out the data we need for peak detection
    emg = data['emg']
    emg_filt = data['emg_filt']
    emg_raw = emg.copy()
    burst_idx = data['spike_idx']
    window = round(data['params']['emg_window'] / 2)

    # try an additional filtering?
    if filt_flag:
        wbf = data['wbf']
        flying_idx = data['flying_idx']
        fs = data['sampling_freq']

        emg = filter_for_spike_detection(emg, fs, wbf, flying_idx=flying_idx)

    # get array of realigned spikes
    _, realigned_idx = realign_spikes(data['spikes'], burst_idx, emg_raw, thresh_factor=0.25)
    bursts_realigned = [emg[slice(idx - window, idx + 2 * window)] for idx in realigned_idx]
    burst_array = np.vstack(bursts_realigned)
    # subtract off initial value
    burst_array -= np.reshape(np.mean(burst_array[:, :window], axis=1), (-1, 1))

    # get array of LOWPASS FILTERED realigned spikes
    bursts_filt_realigned = [emg_filt[slice(idx - window, idx + 2 * window)] for idx in realigned_idx]
    burst_filt_array = np.vstack(bursts_filt_realigned)
    # subtract off initial value
    burst_filt_array -= np.reshape(np.mean(burst_filt_array[:, :window], axis=1), (-1, 1))

    # loop over bursts and count spikes
    n_spikes_list = list()
    spikes_list = list()
    ignore_idx = np.zeros(burst_array.shape[0], dtype=bool)  # store which bursts have no spikes(?)

    for jth, burst in enumerate(burst_array):
        # get filtered version of current spike
        burst_filt = burst_filt_array[jth, :]

        # detect spikes
        spike_idx = detect_burst_peaks_deriv(burst,
                                             burst_waveform_filt=burst_filt,
                                             viz_flag=False)  # )

        # store info - to ignore if we don't detect anything, otherwise spikes
        if spike_idx is None:
            ignore_idx[jth] = True
        else:
            n_spikes_list.append(spike_idx.size)
            spikes_list.append(spike_idx)

    # remove bursts with no spikes
    burst_array = burst_array[~ignore_idx, :]

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------
    # figure out dimensions for plot grid
    n_spikes = np.asarray(n_spikes_list)
    n_spikes_unique = np.unique(n_spikes)

    n_rows = int(np.ceil((n_spikes_unique.size + 1) / n_cols))
    fig_height = 4*n_rows

    fig, ax_list = plt.subplots(n_rows, n_cols, figsize=(10, fig_height))
    ax_list = ax_list.ravel()

    # loop over values of spike num and plot traces
    for ith, spike_num in enumerate(n_spikes_unique):
        # get data for current spike count
        burst_idx_curr = (n_spikes == spike_num)
        bursts = burst_array[burst_idx_curr, :]

        # plot each burst
        for burst in bursts:
            ax_list[ith].plot(burst, 'g-', lw=0.2, alpha=0.2)

        # plot mean
        ax_list[ith].plot(np.mean(bursts, axis=0), 'r-')

        # label axes
        ax_list[ith].set_xticks([])
        ax_list[ith].set_yticks([])
        ax_list[ith].set_title(f'{spike_num} spikes')

    # make the next axis show the histogram of peak times
    ax_list[ith + 1].hist(np.hstack(spikes_list), bins=100)
    ax_list[ith + 1].set_xlabel('spike index')
    ax_list[ith + 1].set_ylabel('count')

    # remove other axes
    for kth in range(ith+2, len(ax_list)):
        ax_list[kth].remove()

    # show figure
    fig.tight_layout()
    plt.show()

    # return
    return fig, ax_list, n_spikes


# ---------------------------------------------------------------------------------
def get_mean_waveform_spikes():
    """
    Function to detect the spike peaks in the MEAN waveform of EMG bursts.
    Because we expect the mean to be smoother than an individual trace, and
    bursts have pretty regular shapes, can use this to constrain general
    peak detection

    """


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # for now, use this as burst detection check
    data_folder = '28_20240529'  # '24_20240520'  # '23_20240517'  #
    axo_num = 2  # 8  #  6  #

    data = load_processed_data(data_folder, axo_num)

    f, al, n_spikes = peak_detection_test(data, filt_flag=True)
