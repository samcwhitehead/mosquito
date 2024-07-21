"""
Code to analyze EMG bursts (e.g. to detect the number of spikes per burst)

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
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
# size of window for realigned bursts
BURST_WINDOW = 1024  # was 1028? # number of time points to get when collecting spike windows

# filter parameters for filter_for_spike_detection
FILT_FLAG = True
FILT_ORDER = 25
MOVING_AVG_WIN = 7

# parameters for spike detection
MIN_HEIGHT_FACTOR = 0.3  # number to multiply by burst height to get minimum spike height
MIN_PROM_FACTOR = 0.025  # number to multiply by burst height to get minimum spike prominence
MIN_DISTANCE = 40  # minimum distance between detected spikes (in index)

DERIV_HEIGHT_FACTOR = 0.1  # number to multiply by burst derivative height to get min peak height
DERIV_WIDTH_BOUNDS = (20, 100)  # range of acceptable *derivative* peak widths, in index

DETREND_FLAG = True  # detrend burst signal?
REFINE_FLAG = True  # try to refine detected spike peaks (in _deriv method)

# parameters for moving_slope derivative of burst waveform (see util.moving_slope)
SUPPORTLENGTH = 101
MODELORDER = 7

# parameters for realigning spikes
THRESH_FACTOR = 0.4  # number to multiply by burst height to get threshold for rise time point


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
# ---------------------------------------------------------------------------------
def filter_for_spike_detection(emg, fs, wbf, flying_idx=None,
                               filt_order=FILT_ORDER,
                               moving_avg_win=MOVING_AVG_WIN):
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
def detect_burst_peaks(burst_waveform, min_height_factor=MIN_HEIGHT_FACTOR,
                       min_prom_factor=MIN_PROM_FACTOR, min_distance=MIN_DISTANCE,
                       burst_waveform_filt=None, detrend_flag=DETREND_FLAG,
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
def detect_burst_peaks_deriv(burst_waveform,
                             deriv_height_factor=DERIV_HEIGHT_FACTOR,
                             min_height_factor=MIN_HEIGHT_FACTOR,
                             deriv_width_bounds=DERIV_WIDTH_BOUNDS,
                             burst_waveform_filt=None,
                             supportlength=SUPPORTLENGTH,
                             modelorder=MODELORDER,
                             refine_flag=REFINE_FLAG,
                             detrend_flag=DETREND_FLAG,
                             viz_flag=False):
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
        supportlength, modelorder: parameters for moving_slope derivative function
            (see util.moving_slope)
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
    burst_waveform_dot = moving_slope(burst_waveform,
                                      supportlength=supportlength,
                                      modelorder=modelorder)
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
def realign_spikes(spikes, spike_idx, emg, window=BURST_WINDOW,
                   thresh_factor=THRESH_FACTOR, viz_flag=False):
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
        spike = signal.detrend(spike)
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
def run_spike_detection(data,
                        deriv_height_factor=DERIV_HEIGHT_FACTOR,
                        min_height_factor=MIN_HEIGHT_FACTOR,
                        deriv_width_bounds=DERIV_WIDTH_BOUNDS,
                        supportlength=SUPPORTLENGTH,
                        modelorder=MODELORDER,
                        realign_thresh=THRESH_FACTOR,
                        filt_order=FILT_ORDER,
                        moving_avg_win=MOVING_AVG_WIN,
                        refine_flag=REFINE_FLAG,
                        detrend_flag=DETREND_FLAG,
                        filt_flag=FILT_FLAG,
                        viz_flag=False):
    """
    Function to perform spike detection on all bursts within a trial, taking a
    data dictionary as input

    Args:
        data: data dictionary from process_abf.py
        deriv_height_factor: number to multiply by burst waveform derivative height
            to get minimum height in peak detection
        min_height_factor: number to multiply by burst_waveform height
            to get minimum height for peaks
        deriv_width_bounds: min and max *derivative* peak width, in index units
        supportlength, modelorder: parameters for moving_slope derivative function
            (see util.moving_slope)
        realign_thresh: factor used to find rise time in realign_spikes
        filt_order: order for bandstop filter in filter_emg_for_spike_detection
        moving_avg_win: rolling mean window size in filter_emg_for_spike_detection
        refine_flag: try an additional refinement step for peak location at the end?
        detrend_flag: detrend emg signal prior to spike detection?
        filt_flag: try an additional filtering of data to remove wbf noise?
        viz_flag: bool, visualize spike detection?

    Returns:


    TODO:
        - add spike props?
    """
    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------
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

        emg = filter_for_spike_detection(emg, fs, wbf,
                                         flying_idx=flying_idx,
                                         filt_order=filt_order,
                                         moving_avg_win=moving_avg_win)

    # realign bursts to rise time
    _, realigned_idx = realign_spikes(data['spikes'], burst_idx, emg_raw,
                                      window=window, thresh_factor=realign_thresh)

    # get bursts into zeroed array
    bursts_realigned = [emg[slice(idx - window, idx + 2 * window)] for idx in realigned_idx]
    burst_array = np.vstack(bursts_realigned)
    # subtract off initial value
    burst_array -= np.reshape(np.mean(burst_array[:, :window], axis=1), (-1, 1))

    # get array of LOWPASS FILTERED realigned spikes
    bursts_filt_realigned = [emg_filt[slice(idx - window, idx + 2 * window)]
                             for idx in realigned_idx]
    burst_filt_array = np.vstack(bursts_filt_realigned)
    # subtract off initial value
    burst_filt_array -= np.reshape(np.mean(burst_filt_array[:, :window], axis=1), (-1, 1))

    # loop over bursts and count spikes (calling them peaks for now to avoid confusion)
    n_peaks_list = list()
    peak_idx_list = list()
    peak_idx_global_list = list()
    burst_idx_list = list()
    spike_num_list = list()
    ignore_idx = np.zeros(burst_array.shape[0], dtype=bool)  # store which bursts have no spikes(?)

    for jth, burst in enumerate(burst_array):
        # get filtered version of current spike
        burst_filt = burst_filt_array[jth, :]

        # detect spikes
        peak_idx = detect_burst_peaks_deriv(burst,
                                            deriv_height_factor=deriv_height_factor,
                                            min_height_factor=min_height_factor,
                                            deriv_width_bounds=deriv_width_bounds,
                                            burst_waveform_filt=burst_filt,
                                            supportlength=supportlength,
                                            modelorder=modelorder,
                                            refine_flag=refine_flag,
                                            detrend_flag=detrend_flag,
                                            viz_flag=False)

        # if we don't detect anything, note the index and put some filler values in
        if peak_idx is None:
            ignore_idx[jth] = True
            peak_idx = np.array([np.nan])
            n_peaks_curr = 0
            spike_nums = np.array([np.nan])
        else:
            n_peaks_curr = peak_idx.size
            spike_nums = np.arange(n_peaks_curr)
        # strore info
        n_peaks_list.append(n_peaks_curr)
        peak_idx_list.append(peak_idx)
        peak_idx_global_list.append(peak_idx + (realigned_idx[jth] - window))
        burst_idx_list.append([realigned_idx[jth] for x in peak_idx])
        spike_num_list.append(spike_nums)

    # put spike info into array form
    peak_idx_array = np.hstack(peak_idx_list)
    peak_idx_global_array = np.hstack(peak_idx_global_list)
    burst_idx_array = np.hstack(burst_idx_list)
    spike_num_array = np.hstack(spike_num_list)

    # put spike info into pandas dataframe
    peaks_df = pd.DataFrame.from_dict({'peak_idx': peak_idx_array,
                                       'peak_idx_global': peak_idx_global_array,
                                       'burst_idx': burst_idx_array,
                                       'peak_num': spike_num_array})

    # ---------------------------------------------------------
    # Plotting?
    # ---------------------------------------------------------
    if viz_flag:
        # for visualizations, remove problem bursts
        n_spikes = np.asarray(n_peaks_list)
        n_spikes_cleaned = n_spikes[~ignore_idx]
        n_spikes_unique = np.unique(n_spikes_cleaned)
        burst_array_cleaned = burst_array[~ignore_idx, :]

        # figure out dimensions for plot grid
        n_cols = 3
        n_rows = int(np.ceil((n_spikes_unique.size + 1) / n_cols))
        fig_height = 4*n_rows

        fig, ax_list = plt.subplots(n_rows, n_cols, figsize=(10, fig_height))
        ax_list = ax_list.ravel()

        # loop over values of spike num and plot traces
        for ith, spike_num in enumerate(n_spikes_unique):
            # get data for current spike count
            burst_idx_curr = (n_spikes_cleaned == spike_num)
            bursts = burst_array_cleaned[burst_idx_curr, :]

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
        ax_list[ith + 1].hist(np.hstack(peak_idx_list), bins=100)
        ax_list[ith + 1].set_xlabel('spike index')
        ax_list[ith + 1].set_ylabel('count')

        # remove other axes
        for kth in range(ith+2, len(ax_list)):
            ax_list[kth].remove()

        # show figure
        fig.tight_layout()
        plt.show()

    # return
    return peaks_df


# ---------------------------------------------------------------------------------
def load_burst_data(folder_id, axo_num,
                    root_path='/media/sam/SamData/Mosquitoes',
                    subfolder_str='*_{:04d}',
                    data_file_str='spike_df',
                    ext='.pkl'):
    """
    Convenience function to load a dataframe containing burst info


    Args:
        folder_id: folder containing processed data (in form XX_YYYYMMDD).
            If just a number is given, search for the matching folder index
        axo_num: per-day index of data file
        root_path: parent folder containing set of experiment folders
        subfolder_str: format of folder name inside experiment_folder
        data_file_str: string giving data filename, used to look for file
        ext: extension for analysis file

    Returns: None
    """
    # check input type -- if it's a two-digit number, search for folder
    if str(folder_id).isnumeric():
        expr_folders = [f for f in os.listdir(root_path)
                        if os.path.isdir(os.path.join(root_path, f))
                        and f[:2].isdigit()]
        expr_folder_inds = [int(f.split('_')[0]) for f in expr_folders]
        expr_folder = expr_folders[expr_folder_inds.index(int(folder_id))]
    else:
        expr_folder = folder_id

    # find path to data file, given info
    search_path = os.path.join(root_path, expr_folder, subfolder_str.format(axo_num))
    search_results = glob.glob(os.path.join(search_path, f'*{data_file_str}{ext}'))

    # check that we can find a unique matching file
    if len(search_results) != 1:
        raise ValueError('Could not locate file in {}'.format(search_path))

    data_path_full = search_results[0]

    # load pickled data file
    data = pd.read_pickle(data_path_full)

    return data


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # for now, use this as burst detection check
    data_folder = 23  # '23_20240517'  #
    axo_num = 2  # 8  #  6  #

    # load data
    data = load_processed_data(data_folder, axo_num)

    # get spike info
    spike_df = run_spike_detection(data, viz_flag=True)
