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

# ---------------------------------------
# PARAMS
# ---------------------------------------
BURST_WINDOW = 1028  # number of time points to get when collecting spike windows


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
# ---------------------------------------------------------------------------------
def detect_burst_peaks(emg, t, burst_idx, window=BURST_WINDOW,
                       min_height_factor=0.3, min_prom_factor=0.01,
                       min_distance=8, viz_flag=False):
    """
    Function to detect the peaks (spikes) within a power muscle burst

    Args:
        emg: numpy array containing emg signal
        t: numpy array containing time measurements (same size as 'emg')
        burst_idx: index associated with detected burst
        window: int, size of window around burst_idx
        min_height_factor: number to multiply by burst_waveform height
            to get minimum height in peak detection
        min_prom_factor: number to multiply by burst_waveform height
            to get minimum prominence in peak detection
        min_distance: minimum distance (x direction) between adjacent peaks
        viz_flag: bool, visualize peak detection?

    Returns:
        peaks: a list containing spike indices
        peaks_df: a pandas DataFrame containing spike info

    """
    # get burst waveform and subtract off initial value
    idx_range = np.arange(burst_idx - window, burst_idx + window)
    burst_waveform = emg[idx_range] - emg[idx_range][0]

    # set some parameters for peak detection
    min_height = min_height_factor * np.max(burst_waveform)
    min_prominence = min_prom_factor * np.max(burst_waveform)

    # do peak detection
    peaks, _ = signal.find_peaks(burst_waveform,
                                 distance=min_distance,
                                 height=(min_height, None),
                                 prominence=(min_prominence, None))

    if viz_flag:
        fig, ax = plt.subplots()
        ax.plot(idx_range, burst_waveform)
        ax.plot(idx_range[peaks], burst_waveform[peaks], 'rx')

        ax.set_xlabel('x (index)')
        ax.set_ylabel('emg (V)')

        plt.show()

    if peaks.size < 1:
        print('Could not find peaks')
        return None, None

    # get peaks in index of full 'emg' array
    peaks += idx_range[0]

    # get height from left minima to peak height
    left_heights = list()
    left_heights.append(emg[peaks[0]])

    for ith in np.arange(1, peaks.size):
        left_min = np.min(emg[peaks[ith-1]:peaks[ith]])
        left_height = emg[peaks[ith]] - left_min
        left_heights.append(left_height)

    # create peaks DataFrame
    peaks_dict = dict()
    peaks_dict['burst_idx'] = burst_idx*np.ones(peaks.shape, dtype=int)
    peaks_dict['spike_idx'] = peaks
    peaks_dict['spike_t'] = t[peaks]
    peaks_dict['spike_dt'] = t[peaks] - t[peaks[0]]  # time from first peak
    peaks_dict['spike_num'] = np.arange(peaks.size)
    peaks_dict['spike_height'] = emg[peaks]
    peaks_dict['spike_left_height'] = np.asarray(left_heights)
    peaks_df = pd.DataFrame.from_dict(peaks_dict)

    return peaks, peaks_df


# ---------------------------------------------------------------------------------
def realign_spikes(spikes, spike_idx, emg, window, thresh_factor=0.5,
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
            continue

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


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    print('under construction')
