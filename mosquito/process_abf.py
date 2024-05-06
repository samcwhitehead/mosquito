"""
Code to analyze EMG data collected during mosquito experiments

TODO:
    - function to run analysis and save
    - print out spike detection sanity check
    - add spike clustering functions
    - add high-speed video analysis
    - should this be done with wavelets?

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import pickle
import pyabf
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import signal
from util import iir_notch_filter, butter_bandpass_filter

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# ---------------------------------------
# PARAMS
# ---------------------------------------
# debug?
DEBUG_FLAG = False

# microphone filter params
MIC_LOWCUT = 200  # lower cutoff frequency for mic bandpass filter
MIC_HIGHCUT = 850  # higher cutoff frequency for mic bandpass filter
NPERSEG = 16384  # length of window to use in short-time fourier transform for wbf estimate

# emg filter params - POWER
EMG_LOWCUT_POWER = 1  # 50  # lower cutoff frequency for muscle emg bandpass filter
EMG_HIGHCUT_POWER = 50  # 2000  # higher cutoff frequency for muscle emg bandpass filter
EMG_BTYPE_POWER = 'bandpass'  # butter filter type (bandpass or bandstop)
EMG_WINDOW_POWER = 2048  # number of time points to get when collecting spike windows
EMG_OFFSET_POWER = 256  # peak offset when storing spike windows
THRESH_FACTORS_POWER = (1, 20)  # factors multiplied by thresh in spike peak detection

# emg filter params - STEERING
EMG_LOWCUT_STEER = 50
EMG_HIGHCUT_STEER = 2000
EMG_BTYPE_STEER = 'bandpass'
EMG_WINDOW_STEER = 32
EMG_OFFSET_STEER = 4
THRESH_FACTORS_STEER = (2, 8)

# general emg filter params
NOTCH_Q = 2.0  # quality factor for iir notch filter
MIN_SPIKE_DT = 0.0015  # in seconds


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------

# ---------------------------------------------------------------------------------
# noinspection PyIncorrectDocstring
def define_params(muscle_type,
                  mic_lowcut=MIC_LOWCUT,
                  mic_highcut=MIC_HIGHCUT,
                  nperseg=NPERSEG,
                  notch_q=NOTCH_Q,
                  min_spike_dt=MIN_SPIKE_DT,
                  emg_lowcut_power=EMG_LOWCUT_POWER,
                  emg_highcut_power=EMG_HIGHCUT_POWER,
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
                  ):
    """
    Convenience function to create a dictionary containing different processing
    parameters for power vs steering muscles

    Args:
        muscle_type: 'steer' or 'power'
        * a bunch of params defined above *

    Returns
        params: dictionary of params

    """
    # initialize dict
    params = dict()

    # fill in general params
    params['mic_lowcut'] = mic_lowcut
    params['mic_highcut'] = mic_highcut
    params['nperseg'] = nperseg
    params['notch_q'] = notch_q
    params['min_spike_dt'] = min_spike_dt

    # fill in dict depending on muscle type
    if muscle_type == 'steer':
        params['emg_lowcut'] = emg_lowcut_steer
        params['emg_highcut'] = emg_highcut_steer
        params['emg_btype'] = emg_btype_steer
        params['emg_window'] = emg_window_steer
        params['emg_offset'] = emg_offset_steer
        params['thresh_factors'] = thresh_factors_steer

    elif muscle_type == 'power':
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
    channel_name_dict = {'Microphon':'mic',
                         'EMG':'emg',
                         'CAM':'cam'}
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

    # print header info?
    if print_flag:
        print(abf_dict['header'])

    return abf_dict


# ---------------------------------------------------------------------------------
def filter_microphone(mic, fs, lowcut=MIC_LOWCUT, highcut=MIC_HIGHCUT,
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
def get_wbf(mic, fs, nperseg=NPERSEG, viz_flag=False):
    """
    Get per-timestep estimates of wingbeat frequency from (filtered) mic signal
    using the short-time fourier transform (STFT)

    Args:
        mic: microphone signal
        fs: sampling frequency, in Hz
        nperseg: length of each segment for STFT. input to stft
        viz_flag: bool, visulize wingbeat frequency estimate?

    Returns: wbf, estimated wingbeat frequency

    """
    f, t, Zxx = signal.stft(mic, fs, nperseg=nperseg)

    # get max vals at each time point
    freq_max_idx = np.argmax(np.abs(Zxx), axis=0)
    freq_max = f[freq_max_idx]

    # visualize wbf estimation?
    if viz_flag:
        # make figure
        fig, ax = plt.subplots(figsize=(11, 6))

        # plot data
        # ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        ax.plot(t, freq_max, 'r-')

        # label axes
        # ax.set_title('STFT Magnitude')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

        plt.show()

    # upsample estimated wbf to full time range
    t_full = (1 / fs) * np.arange(mic.size)
    wbf = np.interp(t_full, t, freq_max)

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
def filter_emg(emg, fs, wbf_mean, lowcut=EMG_LOWCUT_POWER,
               highcut=EMG_HIGHCUT_POWER, notch_q=NOTCH_Q,
               band_type='bandstop', viz_flag=False):
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
    emg_filt = iir_notch_filter(emg, wbf_mean, 1/fs, Q=notch_q)  # Q=0.5

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
def detect_spikes(emg, fs, window=EMG_WINDOW_POWER, offset=EMG_OFFSET_POWER,
                  min_spike_dt=MIN_SPIKE_DT, thresh_factors=THRESH_FACTORS_POWER,
                  rm_spikes_flag=True, abs_flag=False, viz_flag=False):
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
        rm_spikes_flag: remove extracted spikes that have non-zero mean in
            the early portion of the time trace? This is particularly
            helpful for removing 'spikes' that are actually just the recovery
            overshoot
        abs_flag: bool, take absolute value for peak detection?
        viz_flag: bool, visualize spike detection?

    Returns:
        spikes: array of emg windows around spikes
        spike_t: time values that should be used for spikes
        spike_idx: indices of spike events in emg data

    """
    # calculate threshold for spike detection using median absolute deviation
    mad = np.median(np.abs(emg - np.median(emg)))

    # threshold is calculated by finding ~4 sigma level (sd approximated by MAD)
    thresh = 5*mad/0.6745  # 0.6745 is the factor relating median absolute deviation to sd

    # do first-pass spike location estimate using peak detection
    min_peak_dist = np.round(fs*min_spike_dt)
    min_height = thresh_factors[0] * thresh  # 2*
    max_height = thresh_factors[1] * thresh  # 8*

    if abs_flag:
        peak_signal = np.abs(emg)
    else:
        peak_signal = emg
    peaks, _ = signal.find_peaks(peak_signal,
                                 height=(min_height, max_height),
                                 distance=min_peak_dist)

    # extract window around detected peaks (+/- spike window)
    n_pts = emg.size
    spike_list = []
    peaks_new = []

    # loop over detected peaks
    for pk in peaks:
        # get range around current peak
        if (pk - window) < 0 or (pk + window) > n_pts:
            continue
        spike_ind_init = range(pk - int(window/2), pk + int(window/2))

        # recenter to max value
        pk_new = spike_ind_init[np.argmax(emg[spike_ind_init])]
        spike_ind = range(pk_new - window + offset, pk_new + window + offset)
        if (pk_new - window) < 0 or (pk_new + window) > n_pts:
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

    # do removal
    spikes = spikes[~rm_idx, :]
    peaks = peaks[~rm_idx]

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
def cluster_spikes(spikes, pca_n_comp=20, cluster_num=None):
    """
    Cluster spike waveforms

    Args:
        spikes: NxM array containing spike waveforms, where N is number of spikes
            and M is number of time points
        pca_n_comp: number of PCA components to take for clustering
        cluster_num: number of clusters to use. if None, try to evaluate optimal
            cluster number using silhouette scores


    Returns:

    """
    # do PCA on spike waveforms to make clustering less expensive
    pca = PCA(n_components=pca_n_comp)
    pca.fit(spikes)
    pca_result = pca.transform(spikes)

    # try evaluating cluster numbers?
    if cluster_num is None:
        # initialize lists
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
        silhouette_avgs = list()

        # initialize plotting tools
        fig, ax_list = plt.subplots(2, 4, figsize=(11, 9))
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
                ax_list[ith].plot(1000 * spikes, spike, color=col, lw=0.35)

            ax_list[ith].autoscale(enable=True, axis='x', tight=True)
            ax_list[ith].set_xlabel('time (ms)', fontsize=20);
            ax_list[ith].set_ylabel('emg (V)', fontsize=20);
            ax_list[ith].set_title('clustering for n={}'.format(n_clusters), fontsize=23)

        # add final plot for silhouette scores
        ax_list[-1].plot(range_n_clusters, silhouette_avgs)
        ax_list[-1].set_xlabel('Number of clusters', fontsize=20)
        ax_list[-1].set_ylabel('Silhouette score', fontsize=20)
        ax_list[-1].set_title('Silhouette score vs cluster count')

        plt.show()

        # get cluster number from scores
        cluster_num = range_n_clusters[np.argmax(silhouette_avgs)]

        # save cluster evaluation results


    # do clustering

# ---------------------------------------------------------------------------------
def estimate_spike_rate(spike_idx, fs, n_pts, viz_flag=False):
    """
    Function to take in spike times as discrete events and output continuous
    spike rate

    Args:
        spike_idx: indices of spikes in time (subscripts)
        fs: sampling frequency, in Hz
        n_pts: size of full time series arrays
        viz_flag: bool, visualize spike rate estimate?

    Returns: spike_rate, the estimated spike rate in Hz
    """
    # create gaussian window kernel for convolution
    mean_isi = np.mean(np.diff(spike_idx))  # guess how big we need gaussian to be
    win_std = int(2**np.round(np.log2(mean_isi)))
    win_n_pts = 8*win_std
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
def process_abf(filename, muscle_type, debug_flag=False):
    """
    Combines functions above to do full processing of abf EMG and microphone data

    Args:
        filename: full path to abf file
        muscle_type: string giving muscle recording type ('steer' or 'power')
        debug_flag: bool, visualize processing steps?

    Returns:

    """
    # get parameters for current muscle type
    params = define_params(muscle_type)

    # load abf data
    abf_dict = my_load_abf(filename)

    # ---------------------------------
    # filter and process mic signal
    mic = abf_dict['mic']
    fs = abf_dict['sampling_freq']

    # filter mic
    mic_filt = filter_microphone(mic, fs,
                                 lowcut=params['mic_lowcut'],
                                 highcut=params['mic_highcut'],
                                 viz_flag=debug_flag)

    # estmate wingbeat phase
    mic_phase = estimate_microphone_phase(mic_filt,
                                          viz_flag=debug_flag)

    # get wingbeat frequencies (both mean and per-timestep)
    wbf_mean = get_wbf_mean(mic_filt, fs)
    wbf = get_wbf(mic_filt, fs,
                  nperseg=params['nperseg'],
                  viz_flag=debug_flag)

    # add newly calculated stuff to dict
    abf_dict['mic_filt'] = mic_filt
    abf_dict['mic_phase'] = mic_phase
    abf_dict['wbf_mean'] = wbf_mean
    abf_dict['wbf'] = wbf

    # ---------------------------------
    # filter and process emg signal
    emg = abf_dict['emg']

    # filter emg
    emg_filt = filter_emg(emg, fs, wbf_mean,
                          lowcut=params['emg_lowcut'],
                          highcut=params['emg_highcut'],
                          notch_q=params['notch_q'],
                          band_type=params['emg_btype'],
                          viz_flag=debug_flag)

    # detect spikes
    abs_flag = (muscle_type != 'power')  # if looking at power muscles, only look at positive peaks
    spikes, spike_t, spike_idx = detect_spikes(emg_filt, fs,
                                               window=params['emg_window'],
                                               offset=params['emg_offset'],
                                               min_spike_dt=params['min_spike_dt'],
                                               thresh_factors=params['thresh_factors'],
                                               abs_flag=abs_flag,
                                               viz_flag=True)  # DEBUG_FLAG

    # cluster spikes(?)
    # TODO: write function and add

    # estimate spike rate
    # TODO: incorporate clustered spikes into this
    spike_rate = estimate_spike_rate(spike_idx, fs, emg.size,
                                     viz_flag=True)

    # add emg content to dictionary
    abf_dict['emg_filt'] = emg_filt
    abf_dict['spikes'] = spikes
    abf_dict['spike_t'] = spike_t
    abf_dict['spike_idx'] = spike_idx
    abf_dict['spike_rate'] = spike_rate

    # also add params to dictionary
    abf_dict['params'] = params

    # return
    return abf_dict


# ---------------------------------------------------------------------------------
def save_processed_data(filename, abf_dict, file_type='pkl'):
    """
    Convenience function to save dictionary containing processed abf data to disk

    Args:
        filename: full path to save to
        abf_dict: dictionary containing abf data
        file_type: method for saving data. Currently, 'h5' or 'pkl'

    Returns: None
    """
    # remove extension from filepath, if we have it
    path, ext = os.path.splitext(filename)
    savepath = path + '.{}'.format(file_type)

    # save data
    if file_type == 'h5':
        with h5py.File(savepath, 'w') as f:
            for key, entry in abf_dict.items():
                if isinstance(entry, dict):
                    for kkey, eentry in entry.items():
                        f.create_dataset('{}/{}'.format(key, kkey),
                                         data=eentry)
                else:
                    f.create_dataset(key, data=entry)

    elif file_type == 'pkl':
        pickle.dump(abf_dict, open(savepath, "wb"))

    else:
        print('file type {} not currently supported'.format(file_type))

    return None

#
# # ---------------------------------------------------------------------------------
# def run_abf_analysis():
#     """
#     Function to do full analysis of abf file and save results
#
#
#     """


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------
    # path to data file
    data_root = '/media/sam/SamData/HighSpeedVideo/Mosquitoes'
    data_folder = '13_20240426'
    axo_num_list = [12]  # np.arange(5,14)

    for axo_num in axo_num_list:
        data_path = os.path.join(data_root, data_folder,
                                 'axo_recording_{:02d}'.format(axo_num))
        abf_name = [fn for fn in os.listdir(data_path) if fn.endswith('.abf')][0]
        abf_path = os.path.join(data_path, abf_name)

        print('Current data file: \n', abf_path)

        # -----------------------------------------------------------
        # get muscle type for current mosquito
        log_path = os.path.join(data_root, 'experiment_log.xlsx')
        log_df = pd.read_excel(log_path)

        row_idx = (log_df['Day'] == data_folder) & (log_df['Axo Num'] == axo_num)
        muscle_type = log_df.loc[row_idx]['Target Muscle'].values[0]

        if not muscle_type == 'power':
            muscle_type = 'steer'

        # -----------------------------------------------------------
        # try running analysis
        data = process_abf(abf_path, muscle_type, debug_flag=DEBUG_FLAG)

        # -----------------------------------------------------------
        # save dictionary containing processed data
        save_name = abf_name.replace('.abf', '_processed.p')
        save_path = os.path.join(data_path, save_name)
        save_processed_data(save_path, data, file_type='pkl')

        print('Completed saving: \n {}'.format(save_path))
