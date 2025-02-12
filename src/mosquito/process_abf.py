"""
Code to analyze EMG data collected during mosquito experiments

TODO:
    - Finish interactive spike detection!
    - should this be done with wavelets?
    - improved detection of flight bouts (scale microphone)

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
import pickle
import pyabf
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pathlib import Path
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.widgets import Button, Slider

try:
    from util import iir_notch_filter, butter_bandpass_filter, idx_by_thresh
except ModuleNotFoundError:
    from .util import iir_notch_filter, butter_bandpass_filter, idx_by_thresh

# ---------------------------------------
# PARAMS
# ---------------------------------------
# debug?
DEBUG_FLAG = True

# when analyzing files, reuse parameters if we've previously analyzed a file?
REUSE_PARAMS_FLAG = False

# file suffix and type to use for saving
FILE_SUFFIX = '_processed'
SAVE_FILE_EXT = '.pkl'

# microphone filter params
MIC_LOWCUT_AEDES = 200  # lower cutoff frequency for mic bandpass filter
MIC_HIGHCUT_AEDES = 850  # higher cutoff frequency for mic bandpass filter
MIC_LOWCUT_DROSOPHILA = 100  # lower cutoff frequency for mic bandpass filter
MIC_HIGHCUT_DROSOPHILA = 300   # higher cutoff frequency for mic bandpass filter
NPERSEG = 16384  # length of window to use in short-time fourier transform for wbf estimate

# flight bout detection (from mic signal)
MIC_RANGE = (0.25, 9.5)  # (0.35, 6.5)  # (0.05, 8.5)  # volts. values outside this range are counted as non-flying
MIN_BOUT_DURATION = 0.5  # 0.25 # seconds. flying bouts must be at least this long
ROLLING_WINDOW = 501  # size of rolling window for mic amplitude processing

# emg filter params - POWER
EMG_LOWCUT_POWER = 1  # 50  # lower cutoff frequency for muscle emg bandpass filter
EMG_HIGHCUT_POWER = 50  # 2000  # higher cutoff frequency for muscle emg bandpass filter
EMG_LOWCUT_POWER_DROSOPHILA = 10  # 10  # 50  # lower cutoff frequency for muscle emg bandpass filter
EMG_HIGHCUT_POWER_DROSOPHILA = 10000  # 2000  # higher cutoff frequency for muscle emg bandpass filter
EMG_BTYPE_POWER = 'bandpass'  # butter filter type (bandpass or bandstop)
EMG_WINDOW_POWER = 2048  # number of time points to get when collecting spike windows
EMG_OFFSET_POWER = 256  # peak offset when storing spike windows
THRESH_FACTORS_POWER = (1.5, 25)  # (0.45, 25)   # (1.5, 15)  # factors multiplied by thresh in spike peak detection

# emg filter params - STEERING
EMG_LOWCUT_STEER = 300  # 550  # 300  # 700
EMG_HIGHCUT_STEER = 10000
EMG_BTYPE_STEER = 'bandpass'
EMG_WINDOW_STEER = 32  # 32
EMG_OFFSET_STEER = 4
THRESH_FACTORS_STEER = (0.55, 4)  # (0.75, 4.0)  # (0.5, 4) # (0.35, 0.7)  # (0.65, 8)

# general spike detection
REMOVE_EDGE_CASE_FLAG = True  # normally True
RECENTER_WINDOW_FACTOR = 0.125/2  # 0.125  # 0.125  # normally 1, but smaller lets us detect small nearby spikes

# general emg filter params
NOTCH_Q = 2.0  # quality factor for iir notch filter
MIN_SPIKE_DT = 0.0005  # 0.0005 # 0.0015  # in seconds


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------

# ---------------------------------------------------------------------------------
# noinspection PyIncorrectDocstring
def define_params(muscle_type,
                  species='aedes',
                  mic_lowcut_aedes=MIC_LOWCUT_AEDES,
                  mic_highcut_aedes=MIC_HIGHCUT_AEDES,
                  mic_lowcut_drosophila=MIC_LOWCUT_DROSOPHILA,
                  mic_highcut_drosophila=MIC_HIGHCUT_DROSOPHILA,
                  nperseg=NPERSEG,
                  notch_q=NOTCH_Q,
                  min_spike_dt=MIN_SPIKE_DT,
                  emg_lowcut_power=EMG_LOWCUT_POWER,
                  emg_highcut_power=EMG_HIGHCUT_POWER,
                  emg_lowcut_power_drosophila=EMG_LOWCUT_POWER_DROSOPHILA,
                  emg_highcut_power_drosophila=EMG_HIGHCUT_POWER_DROSOPHILA,
                  emg_btype_power=EMG_BTYPE_POWER,
                  emg_window_power=EMG_WINDOW_POWER,
                  emg_offset_power=EMG_OFFSET_POWER,
                  thresh_factors_power=THRESH_FACTORS_POWER,
                  emg_lowcut_steer=EMG_LOWCUT_STEER,
                  emg_highcut_steer=EMG_HIGHCUT_STEER,
                  emg_btype_steer=EMG_BTYPE_STEER,
                  emg_window_steer=EMG_WINDOW_STEER,
                  emg_offset_steer=EMG_OFFSET_STEER,
                  thresh_factors_steer=THRESH_FACTORS_STEER,
                  remove_edge_case_flag=REMOVE_EDGE_CASE_FLAG,
                  recenter_window_factor=RECENTER_WINDOW_FACTOR,
                  ):
    """
    Convenience function to create a dictionary containing different processing
    parameters for power vs steering muscles

    Args:
        muscle_type: 'steer' or 'power'
        species: 'aedes' or 'drosophila'
        * a bunch of params defined above *

    Returns
        params: dictionary of params

    """
    # initialize dict
    params = dict()

    # fill in general params
    params['nperseg'] = nperseg
    params['notch_q'] = notch_q
    params['min_spike_dt'] = min_spike_dt
    params['recenter_window_factor'] = recenter_window_factor
    params['remove_edge_case_flag'] = remove_edge_case_flag

    # get mic parameters based on species
    if species == 'drosophila':
        params['mic_lowcut'] = mic_lowcut_drosophila
        params['mic_highcut'] = mic_highcut_drosophila
    else:
        params['mic_lowcut'] = mic_lowcut_aedes
        params['mic_highcut'] = mic_highcut_aedes

    # fill in dict depending on muscle type
    if muscle_type == 'steer':
        params['emg_lowcut'] = emg_lowcut_steer
        params['emg_highcut'] = emg_highcut_steer
        params['emg_btype'] = emg_btype_steer
        params['emg_window'] = emg_window_steer
        params['emg_offset'] = emg_offset_steer
        params['thresh_factors'] = thresh_factors_steer

    elif muscle_type == 'power':
        if species == 'drosophila':
            params['emg_lowcut'] = emg_lowcut_power_drosophila
            params['emg_highcut'] = emg_highcut_power_drosophila
        else:
            params['emg_lowcut'] = emg_lowcut_power
            params['emg_highcut'] = emg_highcut_power
        params['emg_btype'] = emg_btype_power
        params['emg_window'] = emg_window_power
        params['emg_offset'] = emg_offset_power
        params['thresh_factors'] = thresh_factors_power

    else:
        print('unknown muscle type provided to define_params')

    return params


# ---------------------------------------------------------------------------------
def my_load_abf(filename, print_flag=False):
    """
    Helper function to load ABF file (containing signals from Digidata) and put
    out a dictionary with useful info.

    Args:
        filename: full path to ABF file
        print_flag: bool, print header info upon loading?

    Returns: abf_dict, a dict containing abf data and info
    """
    # load abf object
    abf = pyabf.ABF(filename)

    # create dict to store abf info
    abf_dict = dict()

    # grab info from abf
    abf_dict['sampling_freq'] = abf.sampleRate  # sampling frequency
    abf_dict['filepath'] = abf.abfFilePath
    abf_dict['units'] = abf.adcUnits
    abf_dict['header'] = abf.headerText

    channel_names = abf.adcNames
    channel_name_dict = {'Microphon': 'mic',
                         'EMG': 'emg',
                         'CAM': 'cam',
                         'ODOR': 'odor',
                         'EMG2': 'emg2'}

    # grab channel data
    for ith, name in enumerate(channel_names):
        # set channel
        abf.setSweep(sweepNumber=0, channel=ith)

        # get time once
        if ith == 0:
            abf_dict['time'] = abf.sweepX

        # get current channel data
        if name in channel_name_dict:
            abf_dict[channel_name_dict[name]] = abf.sweepY
        else:
            abf_dict[name] = abf.sweepY

    # if we have multiple emg signals, put those in a list under a single key
    emg_keys = [key for key in abf_dict.keys() if 'emg' in key]
    if len(emg_keys) > 1:
        # group into list and put under 'emg' key
        emg_list = [abf_dict[kkey] for kkey in emg_keys]
        abf_dict['emg'] = emg_list

        # remove other entries
        remove_keys = [k for k in emg_keys if not k=='emg']
        for rm_key in remove_keys:
            del abf_dict[rm_key]

    # print header info?
    if print_flag:
        print(abf_dict['header'])

    return abf_dict


# ---------------------------------------------------------------------------------
def filter_microphone(mic, fs, lowcut=MIC_LOWCUT_AEDES, highcut=MIC_HIGHCUT_AEDES,
                      viz_flag=False):
    """
    Filter microphone signal using butter bandpass

    Args:
        mic: microphone time series signal
        fs: sampling frequency, in Hz
        lowcut: lower cutoff frequency for bandpass filter, in Hz
        highcut: higher cutoff frequency for bandpass filter, in Hz
        viz_flag: bool, visualize filtering?

    Returns: mic_filt, the filtered microphone signal
    """
    # filter mic signal
    mic_filt = butter_bandpass_filter(mic, lowcut, highcut, 1/fs)

    # visualize filtering?
    if viz_flag:
        # make plot
        fig, ax = plt.subplots(figsize=(11, 5))

        t = (1 / fs) * np.arange(mic.size)
        ax.plot(t, mic, label='raw')
        ax.plot(t, mic_filt, label='filt')

        # label axes
        ax.set_title('filtered microphone signal')
        ax.set_ylabel('microphone (V)')
        ax.set_xlabel('index')

        plt.show()

    return mic_filt


# ---------------------------------------------------------------------------------
def detect_flight_bouts(mic, fs, rolling_window=ROLLING_WINDOW,
                        mic_range=MIC_RANGE, min_bout_duration=MIN_BOUT_DURATION,
                        viz_flag=False):
    """
    Try to determine when the fly is actually flying, based on microphone data

    Args:
        mic: microphone signal (should prob be filtered)
        fs: sampling frequency in Hz
        rolling_window: size of rolling window to use for filtering envelope
        mic_range: lower and upper bounds for mic signal amplitude for
            flight periods.
        min_bout_duration: minimum duration for a flight bout, in seconds
        viz_flag: bool, visualize phase estimate?

    Returns:
        stopped_idx: index when then fly is not flying

    TODO: make the bounds for valid flight periods adjustable
    """
    # apply Hilbert transform to mic signal to get envelope
    analytic_mic = signal.hilbert(mic)
    mic_envelope = np.abs(analytic_mic)

    # take rolling max of envelope and filter it
    envelope_series = pd.Series(mic_envelope)
    envelope_filt = envelope_series.rolling(window=rolling_window,
                                            center=True).max()
    envelope_filt = envelope_filt.rolling(window=rolling_window,
                                          center=True).median()

    # fill nan values
    envelope_filt.ffill(inplace=True)
    envelope_filt.bfill(inplace=True)

    # find range where mic envelope is within expected bounds for flight
    flying_idx = (envelope_filt > mic_range[0]) & \
                 (envelope_filt < mic_range[1])
    flying_idx = flying_idx.values

    # keep only flight bputs that are sufficiently long
    flight_bouts = idx_by_thresh(flying_idx)
    flight_bouts_keep = [bout for bout in flight_bouts if (1/fs)*len(bout) >
                         min_bout_duration]
    flying_idx_keep = np.zeros_like(flying_idx)
    for bout in flight_bouts_keep:
        idx1 = max([bout[0]-1, 0])
        idx2 = min([bout[-1]+1, flying_idx_keep.size])
        flying_idx_keep[idx1:idx2] = True

    # remove stop bouts that are too short
    stop_bouts = idx_by_thresh(~flying_idx_keep)
    stop_bouts_drop = [bout for bout in stop_bouts if (1 / fs) * len(bout) <
                       min_bout_duration]
    for bout in stop_bouts_drop:
        # adding back in too-short stop bouts.
        # the +3 is due to a quirk of idx_by_thresh
        flying_idx_keep[(bout[0] - 1):(bout[-1] + 2)] = True

    # visualize?
    if viz_flag:
        # initialize figure/axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # define indexed time
        t = (1/fs)*np.arange(mic.size)

        # plot data
        mic_flying = mic.copy()
        mic_flying[~flying_idx_keep] = np.nan

        ax.plot(t, mic, 'b-', label='not flying')
        ax.plot(t, mic_flying, 'g-', label='flying')
        ax.plot(t, flying_idx_keep, 'r-')

        # label
        ax.set_xlabel('time (idx)')
        ax.set_ylabel('mic (V)')
        # plt.legend()

        plt.show()

    return flying_idx_keep


# ---------------------------------------------------------------------------------
def get_wbf_mean(mic, fs, wbf_est_high=900):
    """
    helper function to get mean wingbeat frequency using mean difference between
    peaks

    Args:
        mic: microphone signal
        fs: sampling frequency, in Hz
        wbf_est_high: upper bound guess for wbf; used for peak detection

    Returns: wbf_mean, the mean wingbeat frequency
    """
    # get minimum distance between peaks for mic signal
    min_peak_dist = np.round(fs / wbf_est_high)

    # find peaks in mic signal
    peaks, _ = signal.find_peaks(mic, distance=min_peak_dist)

    # get wbf estimate from peak locations
    peak_diff = np.diff(peaks)
    dt_mean = (1/fs)*np.mean(peak_diff)
    wbf_mean = 1/dt_mean
    return wbf_mean


# ---------------------------------------------------------------------------------
def get_wbf(mic, fs, nperseg=NPERSEG, max_wbf=None, viz_flag=False):
    """
    Get per-timestep estimates of wingbeat frequency from (filtered) mic signal
    using the short-time fourier transform (STFT)

    Args:
        mic: microphone signal
        fs: sampling frequency, in Hz
        nperseg: length of each segment for STFT. input to stft
        max_wbf: max frequency that we will consider, in Hz.
            If None, don't set a maximum
        viz_flag: bool, visulize wingbeat frequency estimate?

    Returns: wbf, estimated wingbeat frequency

    """
    # calculate short term fourier transform
    freq, t, Zxx = signal.stft(mic, fs, nperseg=nperseg)

    # restrict attention to frequencies < max_freq
    if max_wbf is not None:
        keep_idx = (freq <= max_wbf)
        freq = freq[keep_idx]
        Zxx = Zxx[keep_idx, :]

    # get max vals at each time point
    freq_max_idx = np.argmax(np.abs(Zxx), axis=0)
    freq_max = freq[freq_max_idx]

    # upsample estimated wbf to full time range
    t_full = (1 / fs) * np.arange(mic.size)
    wbf = np.interp(t_full, t, freq_max)

    # visualize wbf estimation?
    if viz_flag:
        # make figure
        fig, ax = plt.subplots(figsize=(11, 6))

        # plot data
        ax.pcolormesh(t, freq, np.abs(Zxx), shading='gouraud')
        ax.plot(t_full, wbf, 'r-')

        # label axes
        ax.set_title('STFT Magnitude')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

        plt.show()

    # return wingbeat frequency estimate
    return wbf


# ---------------------------------------------------------------------------------
def estimate_microphone_phase(mic, viz_flag=False):
    """
    Calculate an estimate of phase from microphone signal

    Args:
        mic: microphone signal (should prob be filtered)
        viz_flag: bool, visualize phase estimate?

    Returns:
        mic_phase: estimated phase at all time points, radians

    """
    # apply Hilbert transform to mic signal
    analytic_mic = signal.hilbert(mic)
    mic_phase = np.angle(analytic_mic)

    # visualize phase estimate?
    if viz_flag:
        # make figure
        fig, ax = plt.subplots(figsize=(11, 5))

        # plot data
        ax.plot(mic)
        ax.plot(mic_phase)

        # label axes
        ax.set_title('Mic phase estimate')
        ax.set_ylabel('microphone')
        ax.set_xlabel('time (s)')

        plt.show()

    # return phase
    return mic_phase


# ---------------------------------------------------------------------------------
def filter_emg(emg, fs, wbf_mean=None, lowcut=EMG_LOWCUT_POWER,
               highcut=EMG_HIGHCUT_POWER, notch_q=NOTCH_Q,
               band_type='bandpass', viz_flag=False):
    """
    Filter emg voltage signal using butter band + notch filter

    Args:
        emg: emg time series signal
        fs: sampling frequency, in Hz
        wbf_mean: mean wingbeat frequency, in Hz
        lowcut: lower cutoff frequency for bandpass filter, in Hz
        highcut: higher cutoff frequency for bandpass filter, in Hz
        notch_q: quality factor of notch filter
        band_type: type of band filter ('bandpass' or 'bandstop')
        viz_flag: bool, visualize filtering?

    Returns: emg_filt, the filtered emg signal
    """
    # notch filter to remove wbf signal (wbf_mean is cut freq)
    if wbf_mean is not None:
        emg_filt = iir_notch_filter(emg, wbf_mean, 1/fs, Q=notch_q)  # Q=0.5
    else:
        emg_filt = emg.copy()

    # band(pass or stop) filter to remove noise at extremes
    emg_filt = butter_bandpass_filter(emg_filt, lowcut, highcut, 1 / fs,
                                      btype=band_type)

    # visualize filtering?
    if viz_flag:
        # make plot
        fig, ax = plt.subplots(figsize=(11, 5))

        t = (1 / fs) * np.arange(emg.size)
        ax.plot(t, emg, label='raw')
        ax.plot(t, emg_filt, label='filt')

        # label axes
        ax.set_title('filtered emg signal')
        ax.set_ylabel('emg (V)')
        ax.set_xlabel('index')

        plt.show()

    return emg_filt


# ---------------------------------------------------------------------------------
def detrend_emg(emg, window=4*EMG_WINDOW_POWER, viz_flag=False):
    """
    Convenience function for detrending a signal using a rolling median

    This might help remove some of the slow drift
    """
    df = pd.Series(emg)
    emg_rolling_median = df.rolling(window=window, center=True).median()
    emg_rolling_median.fillna(value=0, inplace=True)
    emg_detrend = emg - emg_rolling_median

    # visualize detrending?
    if viz_flag:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(emg, label='original')
        ax.plot(emg_detrend.values, label='detrend')
        ax.legend()
        plt.show()

    return emg_detrend.values


# ---------------------------------------------------------------------------------
def detect_spikes(emg, fs, window=EMG_WINDOW_POWER, offset=EMG_OFFSET_POWER,
                  min_spike_dt=MIN_SPIKE_DT, thresh_factors=THRESH_FACTORS_POWER,
                  recenter_window_factor=RECENTER_WINDOW_FACTOR,
                  detrend_window=4*EMG_WINDOW_POWER, thresh_vals=None,
                  rm_spikes_flag=False, abs_flag=False, viz_flag=False,
                  detrend_flag=False, remove_edge_case_flag=REMOVE_EDGE_CASE_FLAG):
    """
    Detect spikes in EMG data. For now, doing this by setting a threshold
    as a first pass detection method, then extracting windows around spikes
    and aligning to peaks

    Args:
        emg: voltage time series of emg data (prob should be filtered)
        fs: sampling frequency, in Hz
        window: size of window (in index) to take around detected spike
        offset: offset (in index) for spike peak in stored array
        min_spike_dt: minimum time (in seconds) between spikes
        thresh_factors: two-element tuple of multiplicative factors to apply
            to thresh to get upper and lower spike bounds
        recenter_window_factor: what fraction of window should be used to
            search around initially located peak to recenter on spike?
        detrend_window: window size to use if detrending (controlled by
            detrend_flag)
        thresh_vals: two-element tuple giving values for upper and lower
            threshold. Should usually be None, since we estimate thresholds
            on a per-signal basis. However, when manually tuning, it's nice
            to be able to set values.
        rm_spikes_flag: remove extracted spikes that have non-zero mean in
            the early portion of the time trace? This is particularly
            helpful for removing 'spikes' that are actually just the recovery
            overshoot
        abs_flag: bool, take absolute value for peak detection?
        viz_flag: bool, visualize spike detection?
        detrend_flag: try to detrend signal using rolling median?
        remove_edge_case_flag: boolean. remove spike if the maximum in the window
            around peak is at the window's edge? Generally a good idea(?)

    Returns:
        spikes: array of emg windows around spikes
        spike_t: time values that should be used for spikes
        spike_idx: indices of spike events in emg data

    """
    # calculate threshold for spike detection using median absolute deviation
    mad = np.median(np.abs(emg - np.median(emg)))

    # threshold is calculated by finding ~5 sigma level (sd approximated by MAD)
    thresh = 5 * mad / 0.6745  # 0.6745 is the factor relating median absolute deviation to sd

    # do first-pass spike location estimate using peak detection
    min_peak_dist = np.round(fs*min_spike_dt)
    if thresh_vals is None:
        min_height = thresh_factors[0] * thresh  # 2*
        max_height = thresh_factors[1] * thresh  # 8*
    else:
        min_height = thresh_vals[0]
        max_height = thresh_vals[1]

    # detrend data? i.e. try to remove baseline fluctuations
    if detrend_flag:
        peak_signal = detrend_emg(emg, window=detrend_window, viz_flag=False)  # window=window)   #
    else:
        peak_signal = emg.copy()

    # take absolute value of signal (i.e. only look for positive peaks?)
    if abs_flag:
        peak_signal = np.abs(peak_signal)

    peaks, _ = signal.find_peaks(peak_signal,
                                 height=(min_height, max_height),
                                 distance=min_peak_dist)  # , prominence=(0.1, None))

    # extract window around detected peaks (+/- spike window)
    n_pts = emg.size
    spike_list = []
    peaks_new = []
    recenter_window = int(recenter_window_factor * window)

    # loop over detected peaks
    for pk in peaks:
        # get range around current peak
        if (pk - window) < 0 or (pk + window) > n_pts:
            continue

        spike_ind_init = range(pk - recenter_window, pk + recenter_window)  # /4
        # spike_ind_init = range(pk - int(window), pk + int(window))

        # recenter to max value
        pk_new = spike_ind_init[np.argmax(peak_signal[spike_ind_init])]
        # # pk_new = spike_ind_init[np.argmin(emg[spike_ind_init])]
        if ((pk_new == spike_ind_init[0]) or (pk_new == spike_ind_init[-1])) and remove_edge_case_flag:
            # if our recentering takes us to the edge of the window, we're probably not on a peak
            continue
        spike_ind = range(pk_new - window + offset, pk_new + window + offset)
        if (spike_ind[0] < 0) or (spike_ind[-1] >= n_pts):
            # if we're on the edge of the time series, skip
            continue
        spike_waveform = emg[spike_ind]

        # get new window/peak location (but first check if we already have this spike)
        if pk_new not in peaks_new:
            peaks_new.append(pk_new)
            spike_list.append(spike_waveform)

    # update peaks
    peaks = np.array(peaks_new)

    # put in array form
    spikes = np.vstack(spike_list)
    spike_t = np.linspace(-1 * window / fs, window / fs, spikes.shape[1])

    # remove any obviously bad waveforms by removing ones with huge jumps
    spike_max = np.max(np.abs(spikes), axis=1)
    spike_max_median = np.median(spike_max)
    spike_max_mad = 1.4836 * np.median(np.abs(spike_max - spike_max_median))

    pseudo_z = (spike_max - spike_max_median) / spike_max_mad
    rm_idx = np.abs(pseudo_z) > 25  # kind of arbitrary high threshold

    # # do removal
    # spikes = spikes[~rm_idx, :]
    # peaks = peaks[~rm_idx]

    # remove spikes based on first portions of the waveform?
    if rm_spikes_flag:
        init_idx = slice(0, round(window/4))
        init_mean = np.mean(spikes[:, init_idx], axis=1)
        rm_idx = (init_mean < -1 * thresh)

        spikes = spikes[~rm_idx, :]
        peaks = peaks[~rm_idx]

    # visualize spike detection?
    if viz_flag:
        # make figure window
        fig, ax = plt.subplots(figsize=(11, 4))

        # plot data
        t = (1 / fs) * np.arange(emg.size)
        ax.plot(t, emg)
        ax.plot(t[peaks], emg[peaks], 'rx')
        ax.axhline(min_height, color='k', ls='--')
        ax.axhline(-1 * min_height, color='k', ls='--')

        # label axes
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_title('Spike threshold')
        ax.set_ylabel('emg (V)')
        ax.set_xlabel('time (s)')

        plt.show()

    # return spike info
    spike_idx = peaks
    return spikes, spike_t, spike_idx


# ---------------------------------------------------------------------------------
def detect_spikes_interactive(emg, fs, window=EMG_WINDOW_POWER,
                              offset=EMG_OFFSET_POWER, min_spike_dt=MIN_SPIKE_DT,
                              thresh_factors=THRESH_FACTORS_POWER,
                              recenter_window_factor=RECENTER_WINDOW_FACTOR,
                              detrend_window=4*EMG_WINDOW_POWER, thresh_vals=None,
                              rm_spikes_flag=False, abs_flag=False, viz_flag=False,
                              detrend_flag=False,
                              remove_edge_case_flag=REMOVE_EDGE_CASE_FLAG):
    """
    *UNDER CONSTRUCTION*

    Function to generate an interactive plot window that can be used for adjusting
    spike detection parameters

    """
    # calculate threshold for spike detection using median absolute deviation
    mad = np.median(np.abs(emg - np.median(emg)))

    # want a really permissive threshold here, so that the slider has a large range
    thresh = mad / 0.6745  # 0.6745 is the factor relating median absolute deviation to sd

    # do first-pass spike location estimate using peak detection
    min_peak_dist = np.round(fs * min_spike_dt)
    min_height = thresh  # 2*
    max_height = 25*thresh  # 8*

    # detrend data? i.e. try to remove baseline fluctuations
    if detrend_flag:
        peak_signal = detrend_emg(emg, window=detrend_window)  # window=window)   #
    else:
        peak_signal = emg.copy()

    # take absolute value of signal (i.e. only look for positive peaks?)
    if abs_flag:
        peak_signal = np.abs(peak_signal)

    # not really all peaks, but this corresponds to most permissive threshold
    peaks_all, props_all = signal.find_peaks(peak_signal,
                                             height=(min_height, max_height),
                                             distance=min_peak_dist)

    # now set initial value for threshold

    #

# ---------------------------------------------------------------------------------
def cluster_spikes(spikes, spike_t=None, save_path=None, pca_n_comp=20,
                   cluster_num=None, max_clust=8, viz_flag=False):
    """
    Cluster spike waveforms

    Args:
        spikes: NxM array containing spike waveforms, where N is number of spikes
            and M is number of time points
        spike_t: time values for spike window. only used for visualization
        save_path: where to save clustering results
        pca_n_comp: number of PCA components to take for clustering
        cluster_num: number of clusters to use. if None, try to evaluate optimal
            cluster number using silhouette scores
        max_clust: maximum cluster number to check
        viz_flag: bool, visualize clusters in final clustering?

    Returns:
        cluster_labels: labels giving cluster ID for each spike
        cluster_dict: dictionary containing cluster info

    """
    # do PCA on spike waveforms to make clustering less expensive
    pca = PCA(n_components=pca_n_comp)
    pca.fit(spikes)
    pca_result = pca.transform(spikes)

    if spike_t is None:
        spike_t = np.arange(spikes.shape[1])

    # initialize storage for all clusterings
    cluster_dict = dict()

    # try evaluating cluster numbers?
    if cluster_num is None:
        # initialize lists
        range_n_clusters = np.arange(2, max_clust+1)
        silhouette_avgs = list()

        # initialize plotting tools
        n_rows = round(np.ceil(range_n_clusters.size/4))
        fig, ax_list = plt.subplots(n_rows, 4, figsize=(11, 9))
        ax_list = ax_list.ravel()

        for ith, n_clusters in enumerate(range_n_clusters):
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 47 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=47, n_init='auto')
            cluster_labels = clusterer.fit_predict(pca_result)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = silhouette_score(pca_result, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
            silhouette_avgs.append(silhouette_avg)

            # plot clustering in waveform space
            colors = cm.tab10(cluster_labels.astype(float) / n_clusters)
            for spike, col in zip(spikes, colors):
                ax_list[ith].plot(spike_t, spike, color=col, lw=0.35, alpha=0.2)

            ax_list[ith].autoscale(enable=True, axis='x', tight=True)
            # ax_list[ith].set_xlabel('time (ms)', fontsize=20);
            ax_list[ith].set_ylabel('emg (V)', fontsize=12);
            ax_list[ith].set_title('clustering for n={}'.format(n_clusters), fontsize=14)

            # store te current clustering in cluster_dict
            cluster_dict[n_clusters] = cluster_labels

        # add final plot for silhouette scores
        ax_list[-1].plot(range_n_clusters, silhouette_avgs)
        ax_list[-1].set_xlabel('Number of clusters', fontsize=12)
        ax_list[-1].set_ylabel('Silhouette score', fontsize=12)
        ax_list[-1].set_title('Silhouette score vs cluster count')

        fig.tight_layout()

        # get cluster number from scores
        cluster_num = range_n_clusters[np.argmax(silhouette_avgs)]

        # store silhouette scores in cluster_dict
        cluster_dict['silhouette_avgs'] = silhouette_avgs

        # save cluster evaluation results
        if save_path is not None:
            cc = 0
            save_path_full = os.path.join(save_path, f'clustering_check_ch{cc}.png')
            while os.path.exists(save_path_full):
                cc += 1
                save_path_full = os.path.join(save_path, f'clustering_check_ch{cc}.png')

            fig.savefig(save_path_full)

        if viz_flag and save_path is None:
            plt.show()
        else:
            plt.close(fig)

    # do clustering
    clusterer = KMeans(n_clusters=cluster_num, random_state=47, n_init='auto')
    cluster_labels = clusterer.fit_predict(pca_result)

    # put "optimal" cluster number and final clustering in cluster_dict
    cluster_dict['optimal_cluster_num'] = cluster_num
    cluster_dict[cluster_num] = cluster_labels

    # visualize?
    if viz_flag:
        # make figure
        fig, ax = plt.subplots()

        # plot clustering in waveform space
        colors = cm.tab10(cluster_labels.astype(float) / cluster_num)
        for spike, col in zip(spikes, colors):
            ax.plot(spike_t, spike, color=col, lw=0.35)

        # label axes
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_ylabel('emg (V)', fontsize=12);

        # show results
        plt.show()

    # return results
    return cluster_labels, cluster_dict


# ---------------------------------------------------------------------------------
def estimate_spike_rate(spike_idx, fs, n_pts, win_factor=1, viz_flag=False):
    """
    Function to take in spike times as discrete events and output continuous
    spike rate

    Args:
        spike_idx: indices of spikes in time (subscripts)
        fs: sampling frequency, in Hz
        n_pts: size of full time series arrays
        win_factor: integer value to scale gaussian window size and # of pts by
        viz_flag: bool, visualize spike rate estimate?

    Returns: spike_rate, the estimated spike rate in Hz
    """
    # create gaussian window kernel for convolution
    mean_isi = np.mean(np.diff(spike_idx))  # guess how big we need gaussian to be
    win_std = int(win_factor)*int(2**np.ceil(np.log2(mean_isi)))
    win_n_pts = int(win_factor)*8*win_std
    gauss_win = signal.windows.gaussian(win_n_pts, win_std)

    # create logical index array for spikes
    spike_idx_log = np.zeros(n_pts, dtype=bool)
    spike_idx_log[spike_idx] = True

    # perform convolution
    spike_rate = (fs * signal.convolve(spike_idx_log, gauss_win, mode='same')
                  / np.sum(gauss_win))

    # visualize estimate?
    if viz_flag:
        # make fig
        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       figsize=(11, 9),
                                       sharex=True)

        # plot data
        t = (1/fs)*np.arange(n_pts)
        ax1.plot(t, spike_idx_log)
        ax2.plot(t, spike_rate)

        # label axes
        ax1.set_ylabel('spikes')
        ax2.set_ylabel('spike rate')
        ax2.set_xlabel('time (s)')

        plt.show()

    return spike_rate


# ---------------------------------------------------------------------------------
def process_abf(filename, muscle_type, species='aedes', params=None,
                debug_flag=False):
    """
    Combines functions above to do full processing of abf EMG and microphone data

    Args:
        filename: full path to abf file
        muscle_type: string giving muscle recording type ('steer' or 'power')
        species: string giving fly info ('aedes' or 'drosophila')
        params: dictionary containing analysis parameters. If None, use defaults
        debug_flag: bool, visualize processing steps?

    Returns:

    """
    # get parameters for current muscle type
    if params is None:
        params = define_params(muscle_type, species=species)

    # load abf data
    abf_dict = my_load_abf(filename)

    # add general info
    abf_dict['species'] = species
    abf_dict['muscle_type'] = muscle_type
    abf_dict['filename'] = filename

    # define some params that we want to have different values per species
    if species == 'drosophila':
        # maximum wingbeat frequency
        max_wbf = 300

        # window size for detrending emg (if detrend_flag)
        detrend_window = int(params['emg_window'] / 2)

    elif species == 'aedes':
        # maximum wingbeat frequency
        max_wbf = 900

        # window size for detrending emg (if detrend_flag)
        detrend_window = int(params['emg_window'])  # int(4 * params['emg_window'])
    else:
        # not sure what to do for unspecified case
        max_wbf = 900
        detrend_window = int(4 * params['emg_window'])

    # ---------------------------------
    # filter and process mic signal
    mic = abf_dict['mic']
    fs = abf_dict['sampling_freq']

    # filter mic
    mic_filt = filter_microphone(mic, fs,
                                 lowcut=params['mic_lowcut'],
                                 highcut=params['mic_highcut'],
                                 viz_flag=False)

    # estmate wingbeat phase
    mic_phase = estimate_microphone_phase(mic_filt,
                                          viz_flag=False)

    # determine periods of non-flight using mic data
    flying_idx = detect_flight_bouts(mic_filt, fs,
                                     viz_flag=debug_flag)  # debug_flag

    # get wingbeat frequencies (both mean and per-timestep)
    wbf_mean = get_wbf_mean(mic_filt, fs, wbf_est_high=max_wbf)
    wbf = get_wbf(mic_filt, fs,
                  nperseg=params['nperseg'],
                  max_wbf=max_wbf,
                  viz_flag=debug_flag)  # debug_flag

    # add newly calculated stuff to dict
    abf_dict['mic_filt'] = mic_filt
    abf_dict['mic_phase'] = mic_phase
    abf_dict['wbf_mean'] = wbf_mean
    abf_dict['wbf'] = wbf
    abf_dict['flying_idx'] = flying_idx

    # ---------------------------------
    # filter and process emg signal
    # 10/08/24: editing to allow multi-channel processing
    if type(abf_dict['emg']) is not list:
        abf_dict['emg'] = list([abf_dict['emg']])

    # initialize emg content for dictionary
    abf_dict['emg_filt'] = []
    abf_dict['spikes'] = []
    abf_dict['spike_t'] = []
    abf_dict['spike_idx'] = []
    abf_dict['spike_rate'] = []
    abf_dict['cluster_labels'] = []
    abf_dict['cluster_dict'] = []

    for emg in abf_dict['emg']:
        # filter emg
        emg_filt = filter_emg(emg, fs, wbf_mean,
                              lowcut=params['emg_lowcut'],
                              highcut=params['emg_highcut'],
                              notch_q=params['notch_q'],
                              band_type=params['emg_btype'],
                              viz_flag=False)

        # detect spikes
        abs_flag = (muscle_type != 'power')  # if looking at power muscles, only look at positive peaks
        detrend_flag = (muscle_type == 'power')  # if looking at power muscles, detrend signal
        spikes, spike_t, spike_idx = detect_spikes(emg_filt, fs,   #
                                                   window=params['emg_window'],
                                                   offset=params['emg_offset'],
                                                   min_spike_dt=params['min_spike_dt'],
                                                   thresh_factors=params['thresh_factors'],
                                                   recenter_window_factor=params['recenter_window_factor'],
                                                   detrend_window=detrend_window,
                                                   abs_flag=abs_flag,
                                                   detrend_flag=detrend_flag,
                                                   remove_edge_case_flag=params['remove_edge_case_flag'],
                                                   viz_flag=debug_flag)  # DEBUG_FLAG

        # add some of these inputs to params dict for max reproducibility
        params['detrend_window'] = detrend_window
        params['abs_flag'] = abs_flag
        params['detrend_flag'] = detrend_flag

        # cluster spikes(?)
        cluster_labels, cluster_dict = cluster_spikes(spikes,
                                                      spike_t=spike_t,
                                                      save_path=os.path.split(abf_path)[0])

        # estimate spike rate
        # TODO: incorporate clustered spikes into this
        spike_rate = estimate_spike_rate(spike_idx, fs, emg.size,
                                         viz_flag=False)

        # add results for current emg channel to list
        abf_dict['emg_filt'].append(emg_filt)
        abf_dict['spikes'].append(spikes)
        abf_dict['spike_idx'].append(spike_idx)
        abf_dict['spike_t'].append(spike_t)
        abf_dict['spike_rate'].append(spike_rate)
        abf_dict['cluster_labels'].append(cluster_labels)
        abf_dict['cluster_dict'].append(cluster_dict)

    # for backward compatibility, collapse length 1 lists for these entries
    emg_dict_keys = ['emg', 'emg_filt', 'spikes', 'spike_t', 'spike_idx', 'spike_rate', 'cluster_labels']
    for key in emg_dict_keys:
        if len(abf_dict[key]) == 1:
            abf_dict[key] = np.squeeze(np.asarray(abf_dict[key]))

    # also collapse non-array fields if needed
    if len(abf_dict['cluster_dict']) == 1:
        abf_dict['cluster_dict'] = abf_dict['cluster_dict'][0]

    # also add params to dictionary
    abf_dict['params'] = params

    # return
    return abf_dict


# ---------------------------------------------------------------------------------
def reprocess_abf(data_folder, axo_num, data_suffix='_processed',
                  debug_flag=False):
    """
    Function to allow a file to be re-analyzed using the same parameters used
    previously. Hoping this will be helpful as I adjust things for multichannel
    data, as I can rerun analyses without having to fiddle with any parameters

    Args:
        data_folder: folder containing processed data (in form XX_YYYYMMDD).
            If just a number is given, search for the matching folder index
        axo_num: per-day index of data file
        data_suffix: string at the end of processed data file; used to look for
            correct file type
        debug_flag: boolean, passed to process_abf

    Returns:
        data: dictionary containing processed data

    """
    # load previously analyzed file
    data_old = load_processed_data(data_folder, axo_num, data_suffix=data_suffix)

    # read params from data file
    params = data_old['params']

    # get path to .abf file to use for analysis
    if os.path.exists(data_old['filepath']):
        # if the old analysis file properly stored the abf path, use that
        axo_filepath = data_old['filepath']
    else:
        # ... otherwise, need to look for it
        load_filepath = data_old['filepath_load']
        search_path = os.path.split(load_filepath)[0]
        search_results = glob.glob(os.path.join(search_path, '*.abf'))
        if len(search_results) != 1:
            raise ValueError(f'Could not locate .abf file in {search_path}')

        axo_filepath = search_results[0]

    # re-run analyses
    data = process_abf(axo_filepath,
                       data_old['muscle_type'],
                       species=data_old['species'],
                       params=params,
                       debug_flag=debug_flag)

    # return data dict
    return  data


# ---------------------------------------------------------------------------------
def save_processed_data(filename, abf_dict, file_type='.pkl'):
    """
    Convenience function to save dictionary containing processed abf data to disk

    Args:
        filename: full path to save to
        abf_dict: dictionary containing abf data
        file_type: method for saving data. Currently, '.h5' or '.pkl'

    Returns: None
    """
    # remove extension from filepath, if we have it
    path, ext = os.path.splitext(filename)
    savepath = path + f'{file_type}'

    # save data
    if file_type == '.h5':
        with h5py.File(savepath, 'w') as f:
            for key, entry in abf_dict.items():
                if isinstance(entry, dict):
                    for kkey, eentry in entry.items():
                        f.create_dataset('{}/{}'.format(key, kkey),
                                         data=eentry)
                else:
                    f.create_dataset(key, data=entry)

    elif file_type == '.pkl':
        pickle.dump(abf_dict, open(savepath, "wb"))

    else:
        print('file type {} not currently supported'.format(file_type))

    return None


# ---------------------------------------------------------------------------------
def load_processed_data(folder_id, axo_num,
                        root_path='/media/sam/SamData/Mosquitoes',
                        subfolder_str='*_{:04d}',
                        data_suffix='processed',
                        ext='.pkl'):
    """
    Convenience function to LOAD dictionary containing processed abf data

    Args:
        folder_id: folder containing processed data (in form XX_YYYYMMDD).
            If just a number is given, search for the matching folder index
        axo_num: per-day index of data file
        root_path: parent folder containing set of experiment folders
        subfolder_str: format of folder name inside experiment_folder
        data_suffix: string at the end of processed data file; used to look for
            correct file type
        ext: extension for analysis file

    Returns: data, the dictionary containg processed data
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
    search_results = glob.glob(os.path.join(search_path, f'*{data_suffix}{ext}'))

    # check that we can find a unique matching file
    if len(search_results) != 1:
        raise ValueError('Could not locate file in {}'.format(search_path))

    data_path_full = search_results[0]

    # load pickled data file
    data = pickle.load(open(data_path_full, "rb"))
    data['filepath_load'] = data_path_full  # add the full filepath that this was just loaded from

    return data


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------
    # path to data file
    data_root = '/media/sam/SamData/Mosquitoes'
    data_folder = '67_20250205'  # '33_20240626'  # '32_20240625'
    axo_num_list = [8]  #

    # loop over axo files to analyze
    for axo_num in axo_num_list:
        # get path to .abf (axo) file
        data_path = os.path.join(data_root, data_folder,
                                 '*_{:04d}'.format(axo_num))
        abf_search = glob.glob(os.path.join(data_path, '*.abf'))
        if len(abf_search) != 1:
            print('Could not find unique file for axo {}'.format(axo_num))
            continue
        else:
            abf_path = abf_search[0]

        abf_name = Path(abf_path).stem

        print('Current data file: \n', abf_path)

        # -----------------------------------------------------------
        # check if we've already analyzed this folder
        search_name = abf_path.replace('.abf',
                                       f'{FILE_SUFFIX}{SAVE_FILE_EXT}')

        # if we can find an old analysis file, re-use it
        if os.path.exists(search_name) and REUSE_PARAMS_FLAG:
            # re-analyze data
            print(f'Re-analyzing {data_folder}, axo {axo_num} with old params')
            data = reprocess_abf(data_folder, axo_num, data_suffix=FILE_SUFFIX,
                                 debug_flag=DEBUG_FLAG)

            # if we successfully did this, save and continue
            save_processed_data(search_name, data, file_type='pkl')
            continue

        # -----------------------------------------------------------
        # get muscle type for current fly
        log_path = os.path.join(data_root, 'experiment_log.xlsx')
        log_df = pd.read_excel(log_path)

        row_idx = (log_df['Day'] == data_folder) & (log_df['Axo Num'] == axo_num)
        muscle_type = log_df.loc[row_idx]['Target Muscle Type'].values[0]
        muscle_type = muscle_type.strip()
        if not muscle_type == 'power':
            muscle_type = 'steer'

        # -----------------------------------------------------------
        # try to get species for current fly by reading README
        readme_path = os.path.join(data_root, data_folder, 'README.txt')
        if os.path.exists(readme_path):
            # put README information into a dictionary
            readme_dict = dict()
            with open(readme_path) as f:
                for line in f:
                    if line.strip():
                        # read full line
                        line_text = line.rstrip()

                        # parse line into dict item
                        line_text_split = line_text.split(':')
                        key = line_text_split[0]
                        if len(line_text_split) > 1:
                            item = line_text.split(':')[1]
                        else:
                            item = None

                        readme_dict[key] = item

            # get species info. NB: we're going to assume aedes is default
            species_entry = readme_dict['Species']
            drosophila_keywords = ["drosophila", "melanogaster", "hcs+"]

            if any(keyword in species_entry.lower() for keyword in drosophila_keywords):
                species = 'drosophila'
            else:
                species = 'aedes'

        else:
            # otherwise just assume it's a mosquito
            species = 'aedes'

        # -----------------------------------------------------------
        # try running analysis
        data = process_abf(abf_path, muscle_type, species,
                           debug_flag=DEBUG_FLAG)

        # -----------------------------------------------------------
        # save dictionary containing processed data
        save_name = abf_path.replace('.abf', f'{FILE_SUFFIX}{SAVE_FILE_EXT}')
        # save_path = os.path.join(data_path, save_name)
        save_processed_data(save_name, data, file_type=SAVE_FILE_EXT)

        print('Completed saving: \n {}'.format(save_name))
