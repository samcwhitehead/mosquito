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
EMG_WINDOW_POWER = 2048  # number of time points to get when collecting spike windows


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
# ---------------------------------------------------------------------------------
def detect_burst_peaks(emg, t, burst_idx, window=EMG_WINDOW_POWER, min_height_factor=0.3,
                       min_prom_factor=0.01, min_distance=8, viz_flag=False):
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


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    print('under construction')
