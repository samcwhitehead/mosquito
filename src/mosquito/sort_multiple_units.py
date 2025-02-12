"""
Code to extract different units from a single recording.

TODO:
    - function for running full analysis on a file

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.signal import find_peaks, hilbert

from mosquito.process_abf import (load_processed_data, cluster_spikes, save_processed_data, detect_spikes,
                                  estimate_spike_rate, detrend_emg, filter_emg, SAVE_FILE_EXT)


# ---------------------------------------
# PARAMS
# ---------------------------------------
# save data?
SAVE_FLAG = True
SAVE_FILE_EXT = '.pkl'

# region of spike window to use for re-clustering
TRIM_IDX = np.arange(1400, 2100)

# smallest spike rate allowed to be called a unit
MIN_SPIKE_RATE = 0.5  # Hz, will probably depend on species and muscle type!


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------

# ---------------------------------------------------------------------------------
def identify_units(spike_array, spike_idx, n_spikes_min,
                   trim_idx=TRIM_IDX, viz_flag=False):
    """
    Function to try estimating which of the detected spikes are noise and
    which are real units. Further, try to separate the detected spikes into
    different units. We'll mostly do this by looking for waveform clusters
    that have large membership

    Args:
        spike_array: an NxT array of detected spike waveforms, where N is
            the number of spikes and T is the duration of the waveform window
        n_spikes_min: the minimum number of spikes we should expect to see in
            this trial for a valid unit. For a typical power muscle, we would
            guess something like:
                flight_duration = np.sum(flying_idx)/fs ,
                n_spikes_min = np.floor(0.5*flight_duration) ,
            where fs is sampling frequency and flying_idx is the index giving
            flight periods and 0.5 Hz is on the low end of spike rates
        spike_idx: indices giving the location of spikes_array spikes in the
            emg recording
        trim_idx: range of waveform window to keep when re-clustering
        viz_flag: boolean. visualize the waveforms for putative units/noise?

    Returns:
        unit_values: the numbers of the clusters corresponding to units
        unit_spike_idx: list of index arrays, each of which gives the location
            of unit spikes(aka "real" spikes) in the larger data time series
        cluster_dict: dictionary corresponding to new clustering

    """
    # first re-cluster putative spikes using a trimmed version of their
    # waveforms. This reduces the effect of nearby spikes in the window
    _, cluster_dict = cluster_spikes(spike_array[:, trim_idx])

    # take out the optimal clustering (the k number with max silhouette score)
    n_clust = cluster_dict['optimal_cluster_num']
    clustering = cluster_dict[n_clust]

    # get and store the number of members in each cluster
    vals = np.unique(clustering)
    membership = np.asarray([np.sum(clustering == val) for val in vals])

    # take the ones that are above minimum membership. we should also expect them
    # to have ~equal numbers, but there's overlap, etc.
    unit_values = np.where(membership >= n_spikes_min)[0]

    # get indices in each channel where the spikes are occuring
    unit_spike_idx = [spike_idx[np.where(clustering == val)[0]] for val in unit_values]

    # visualize spike identification?
    if viz_flag:
        # define some colors to use for plotting waveforms
        # import matplotlib.cm as cm  # bad practice to have this here
        colors = cm.tab10(np.arange(n_clust) / n_clust)

        # initialize figure
        fig, (ax_units, ax_noise) = plt.subplots(1, 2, figsize=(10, 4),
                                                 sharex=True, sharey=True)

        # plot the clusters we think are units. first loop over units
        for val in unit_values:
            # grab spikes for current unit, loop over them, and plot
            unit_spikes = spike_array[clustering == val, :]
            for spike in unit_spikes:
                ax_units.plot(spike, color=colors[val], lw=0.5, alpha=0.2)

        # plot the clusters we think are noise
        noise_values = np.where(membership < n_spikes_min)[0]
        for val in noise_values:
            noise_spikes = spike_array[clustering == val, :]
            for spike in noise_spikes:
                ax_noise.plot(spike, color=colors[val], lw=0.5, alpha=1.0)

        # display figure
        fig.tight_layout()
        plt.show()

    # return
    return unit_values, unit_spike_idx, cluster_dict


# ---------------------------------------------------------------------------------
def sort_units(unit_spike_idx, emg_signal, unit_values=None):
    """
    Function to sort estimated spikes by magnitude from largest spike to smallest

    NB: I could try to write this to actually reassign cluster numbers, so that
    clustering==0 corresponds to the largest spike, etc. However, don't think that
    is needed for now.

    Args:
        unit_spike_idx: from 'identify_spikes'. List of index arrays for valid unit spikes in
            emg data
        emg_signal: time series array of measured emg signal (probably should use filtered)
        unit_values: from 'identify_spikes'. The cluster values corresponding to valid units

    Returns:
        unit_values_sort: unit values list re-ordered by spike magnitude
        unit_spike_idx_sort: unit spike index list re-ordered by spike magnitude
        unit_magnitudes_sort: sorted list of unit magnitudes
    """
    # get mean spike magnitudes for each unit
    unit_magnitudes = [np.mean(emg_signal[idx]) for idx in unit_spike_idx]

    # get sorting index
    sort_idx = np.argsort(unit_magnitudes)[::-1]

    # reorder
    unit_spike_idx_sort = [unit_spike_idx[ind] for ind in sort_idx]
    if unit_values is not None:
        unit_values_sort = unit_values[sort_idx]
    else:
        unit_values_sort = None

    # also reorder magnitudes to match this new ordering
    unit_magnitudes_sort = [unit_magnitudes[ind] for ind in sort_idx]

    # return
    return unit_spike_idx_sort, unit_magnitudes_sort, unit_values_sort


# ---------------------------------------------------------------------------------
def locate_missing_spike_idx(unit_spike_idx, ref_spike_idx, t):
    """
    Function to find locations in a multi-unit recording where we would expect to find
    (but don't) spikes from the smaller amplitude unit(s). Often the two waveforms
    overlap, and the larger one masks the smaller one. This function tries to guess at
    where those overlaps may be occuring, as well as locate small amplitude spikes that
    may have been missed during the initial spike detection

    The basic idea for this is we'll loop over the spikes of a larger reference unit
    and find the minimum distance between each of those large spikes and a spike from
    the smaller unit. The idea is that each of the power muscles is so regular that if
    we have a long period without a small spike then we're likely missing something

    Args:
        unit_spike_idx: array giving indices for the small unit's spikes in the emg time
            series
        ref_spike_idx:  array giving indices for the larger unit's spikes (used as a
            reference signal) in the emg time series
        t: time measurements for time series data

    Returns:
        missing_idx: indices of reference spikes that correspond to where we think we're
            missing small unit spikes

    """
    # define the factor that we'll multiply by reference spike ISI to determine too-long gaps
    isi_factor = 0.75  # 0.45

    # calculate the minimum distance between each reference spike and all smaller spikes
    min_dist = [np.min(np.abs(t[unit_spike_idx] - t[ref_idx])) for ref_idx in ref_spike_idx]
    min_dist = np.asarray(min_dist)

    # use the gradient to get the local interspike interval
    local_isi = np.gradient(t[ref_spike_idx])

    # missing spike indices are where min_dist goes above some fraction of these isi values
    missing_idx = ref_spike_idx[min_dist > (isi_factor * local_isi)]
    # --------------------------------------

    # return
    return missing_idx


# ---------------------------------------------------------------------------------
def find_small_spikes(missing_spike_idx, small_spike_amp, large_spike_amp,
                      window_size, emg_signal):
    """
    Function to find smaller unit spikes near larger (reference) spikes. This will
     be used to deal with overlapping spike cases.

    First we look for small peaks that are nearby large ones. If we don't find
     anything, we'll just assume they overlap

    Args:
        missing_spike_idx: indices giving the location of the *larger* spikes that
            we think are maybe masking a smaller spike
        small_spike_amp: expected amplitude of smaller unit spikes
        large_spike_amp: expected amplitude of larger unit spikes
        window_size: interval size (in index) to look near reference spike for
            smaller peaks. Typically ~0.5*large_spike_isi
        emg_signal: emg time series data

    Returns:
        new_spike_idx: indices of putative new small spikes
    """
    # initialize storage for new spikes
    new_spike_idx = list()

    # loop over missing spike idx (locations near where we think we're missing a small spike)
    for idx in missing_spike_idx:
        # look for peaks in a window around the missing_spike_idx (location of a larger spike)
        window = np.arange(idx - window_size, idx + window_size + 1)

        # want something that could plausibly be small unit but not as big as large unit
        height_min = 0.5 * small_spike_amp
        height_max = max([1.5 * small_spike_amp, 0.75 * large_spike_amp])
        height_range = (height_min, height_max)

        # do peak finding
        # print(window)
        pks, _ = find_peaks(emg_signal[window], height=height_range)

        # exclude the original (large unit) peak
        window_ctr = np.ceil(0.5 * window.size)
        pks = [pk for pk in pks if not pk == window_ctr]

        # switch case depending on how many peaks we found
        if len(pks) == 1:
            # if we found one, take it as the new spike index.
            new_spike_idx.append(window[pks[0]])

        elif len(pks) > 1:
            # for now, if we find multiple peaks take the one closest to the large reference peak
            # TODO: impove this!
            min_ind = np.argmin(np.abs(np.asarray(pks) - window_ctr))
            new_spike_idx.append(window[pks[min_ind]])

        else:
            # if we find no peaks, assume small spike is being totally masked by the larger one
            # (and return its index)
            new_spike_idx.append(idx)

    # return
    return np.asarray(new_spike_idx)


# ---------------------------------------------------------------------------------
def split_spike_cluster(spike_idx, emg_signal):
    """
    Function to deal with case where k-means clustering groups two units together.
     In this case, we go through and look at the relative heights of nearby
     clusters and assign identity that way.

    NB: this only works to split a cluster into two groups. for 3 units in a
     cluster, I don't have a good solution right now, and it might just be worth
     skipping that data file tbh

    Args:
        spike_idx: indices of spikes in one cluster that we think corresponds to
            two units (indices of those spikes location in the time series data)
        emg_signal: time series of muscle measurements

    Returns:
        unit_spike_idx: 2-element list whose entries are arrays of indices for
            the two units

    """
    # detrend emg signal (this makes local comparisons more accurate)
    emg_detrend = detrend_emg(emg_signal)

    # initialize storage
    large_idx = []  # list for indices of the larger unit
    small_idx = []  # list for indices of the smaller unit

    # loop through spikes
    for ith, ind in enumerate(spike_idx):
        # check neighbors
        if ith == (spike_idx.size - 1):
            right_neighbor = np.nan
        else:
            right_neighbor = emg_detrend[spike_idx[ith + 1]]

        if ith == 0:
            left_neighbor = np.nan
        else:
            left_neighbor = emg_detrend[spike_idx[ith - 1]]

        current_val = emg_detrend[ind]

        if (current_val > right_neighbor) or (current_val > left_neighbor):
            large_idx.append(ind)
        elif (current_val < right_neighbor) and (current_val < left_neighbor):
            small_idx.append(ind)
        else:
            print(f'error placing spike at index {ind} into larger/smaller category')

    # put indices into list and return
    unit_spike_idx = [np.asarray(large_idx), np.asarray(small_idx)]
    return unit_spike_idx


# ---------------------------------------------------------------------------------
def run_sorting_on_file(data, min_spike_rate=MIN_SPIKE_RATE, trim_idx=TRIM_IDX,
                        viz_flag=False):
    """
    Wrapper function to do all the stuff from above on a data file
    (should just be stuff from jupyter notebook)

    Args:
        data: typical data dictionary returned by process_abf.py
        min_spike_rate: guess for the minumum spike rate of a cell, in Hz
        trim_idx: indices in window of spikes to take for re-clustering.
            this is an input to identify_spikes
        viz_flag: bool. if true, generate plots

    Returns:


    TODO: species specific minimum spike rate?
    """
    # ---------------------------
    # Read data
    # ---------------------------
    fs = data['sampling_freq']  # sampling frequency for data, in Hz
    t = data['time']  # time in seconds
    flying_idx = data['flying_idx']  # indices for times steps when fly flies
    emg_filt = data['emg_filt']  # list whose entries are arrays of filtered emg
    spikes = data['spikes']  # list whose entries are arrays of time aligned spikes
    spike_idx = data['spike_idx']  # list whose entries are indices for spike times

    # get the number of channels in the current data file
    n_channels = len(spikes)

    # ---------------------------
    # Identify real units
    # ---------------------------
    # determine the minimum number of spikes we would expect from a real unit
    flight_duration = np.sum(flying_idx) / fs
    n_spikes_threshold = np.floor(min_spike_rate*flight_duration)

    # initialize storage for re-clustered units
    unit_values_list = list()
    unit_spike_idx_list = list()
    cluster_dict_new_list = list()

    # loop over channels and pull out likely real units
    for channel in range(n_channels):
        # estimate spikes corresponding to real units
        unit_values, unit_spike_idx, cluster_dict_new = identify_units(spikes[channel],
                                                                       spike_idx[channel],
                                                                       n_spikes_threshold,
                                                                       trim_idx=trim_idx,
                                                                       viz_flag=viz_flag)

        # store
        unit_values_list.append(unit_values)
        unit_spike_idx_list.append(unit_spike_idx)
        cluster_dict_new_list.append(cluster_dict_new)

    # ---------------------------
    # Split clusters (if needed)
    # ---------------------------
    # get the number of spikes for each of these units. then, we'll look for units with
    # anomalously high spike numbers
    n_spikes_all = [unit_idx.size for unit_spike_idx in unit_spike_idx_list
                    for unit_idx in unit_spike_idx]
    if len(n_spikes_all) == 2:
        n_spikes_baseline = np.min(n_spikes_all)
    else:
        n_spikes_baseline = np.median(n_spikes_all)

    # loop over units and see if any of them need to be/can be split
    for channel in range(n_channels):
        for ith, unit_idx in enumerate(unit_spike_idx_list[channel]):
            # test spike count
            n_spike_check = round(unit_idx.size / n_spikes_baseline)
            if n_spike_check == 2:
                # split cluster
                print(f'Splitting cluster for channel {channel}, unit {ith}')
                unit_idx_new = split_spike_cluster(unit_idx, emg_filt[channel])

                # update unit_spike_idx_list
                unit_spike_idx_list[channel].remove(unit_idx)
                unit_spike_idx_list[channel][ith:ith] = unit_idx_new

                # also update list of unit values (aka cluster values for these units)
                unit_values_list[channel] = np.insert(unit_values_list[channel], ith, ith)

    # ---------------------------
    # Sort units by magnitude
    # ---------------------------
    # initialize some storage
    unit_magnitudes_list = list()
    unit_values_list_new = list()
    unit_spike_idx_list_new = list()

    # loop over channels
    for channel in range(n_channels):
        # read out current data
        unit_values = unit_values_list[channel]
        unit_spike_idx = unit_spike_idx_list[channel]

        # re-order units by magnitude before storing
        if unit_values.size > 1:
            unit_spike_idx, unit_magnitudes, unit_values = sort_units(unit_spike_idx,
                                                                      emg_filt[channel],
                                                                      unit_values)
        else:
            unit_magnitudes = [np.mean(emg_filt[channel][unit_spike_idx[0]])]

        # write data to new lists
        unit_magnitudes_list.append(unit_magnitudes)
        unit_values_list_new.append(unit_values)
        unit_spike_idx_list_new.append(unit_spike_idx)

    # overwrite old lists with new ones
    unit_values_list = unit_values_list_new
    unit_spike_idx_list = unit_spike_idx_list_new

    # ---------------------------
    # Check for small spikes
    # ---------------------------
    # sometimes smaller units are masked by larger ones, which throws off analyses
    ref_ind = 0  # index in lists for channel reference unit (the big one). Should be 0
    unit_spike_idx_list_new = list()

    for channel in range(n_channels):
        # initialize per-channel storage
        unit_spike_idx_new = list()

        # get the indices of the current units (NB: this is still a list of arrays)
        unit_spike_idx = unit_spike_idx_list[channel]

        # get the number of spikes for each unit
        n_spikes = [idx.size for idx in unit_spike_idx]

        # store unit 0 spikes
        unit_spike_idx_new.append(unit_spike_idx[0])

        # check if the smaller unit has many fewer detected spikes than the larger one
        for unit_num in range(1, len(n_spikes)):
            if (n_spikes[0] - n_spikes[unit_num]) > 5:
                # print(f'we need to check channel {channel}, unit {unit_num} for missing spikes')
                # find indices to search around for spikes
                missing_spike_idx = locate_missing_spike_idx(unit_spike_idx[unit_num],
                                                             unit_spike_idx[ref_ind],
                                                             t)

                # guess small unit timing around those holes in the time series
                small_amp = unit_magnitudes_list[channel][unit_num]
                large_amp = unit_magnitudes_list[channel][ref_ind]
                window_size = round(0.5 * np.mean(np.diff(unit_spike_idx[0])))
                new_spike_idx = find_small_spikes(missing_spike_idx,
                                                  small_amp,
                                                  large_amp,
                                                  window_size,
                                                  emg_filt[channel])

                # store results
                new_spike_idx_all = np.sort(np.hstack((new_spike_idx,
                                                       unit_spike_idx[unit_num].copy())))
                unit_spike_idx_new.append(new_spike_idx_all)

            else:
                # otherwise just store a copy of the indices we already have
                unit_spike_idx_new.append(unit_spike_idx[unit_num].copy())

        # append per-channel list to across channel list
        unit_spike_idx_list_new.append(unit_spike_idx_new)

    # visualize the newly located small spikes?
    if viz_flag:
        # initialize figure and axes
        fig, ax_list = plt.subplots(n_channels, 1, figsize=(12, 7), sharex=True)
        ax_list = ax_list.ravel()

        # loop over channels
        for channel in range(n_channels):
            # plot emg signal
            ax_list[channel].plot(t, emg_filt[channel])

            # loop over units
            for unit_num, unit_spike_idx in enumerate(unit_spike_idx_list[channel]):
                # skip the large first unit
                if unit_num == 0:
                    continue

                # plot old spikes for small unit
                ax_list[channel].plot(t[unit_spike_idx],
                                      emg_filt[channel][unit_spike_idx],
                                      marker='x',
                                      linestyle='none',
                                      label='old')

                # plot new spikes for small unit
                ax_list[channel].plot(t[unit_spike_idx_list_new[channel][unit_num]],
                                      emg_filt[channel][unit_spike_idx_list_new[channel][unit_num]],
                                      marker='o',
                                      markerfacecolor='none',
                                      linestyle='none',
                                      label='new')

            # make legend
            ax_list[channel].legend()

        # display plot
        plt.show()

    # ---------------------------
    # Add to data dict
    # ---------------------------
    data['sorted_units'] = unit_spike_idx_list
    data['sorted_units_estimate'] = unit_spike_idx_list_new

    return data


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # TEMP -- try an example data file
    # load data file
    data_folder = 63  # 66 # 65
    axo_num = 7  # 1  # 4

    data = load_processed_data(data_folder, axo_num)

    # run analyses
    data = run_sorting_on_file(data, viz_flag=True)

    # save results?
    if SAVE_FLAG:
        current_path = data['filepath_load']
        save_path, _ = os.path.splitext(current_path)
        save_path += '_sort' + SAVE_FILE_EXT
        save_processed_data(save_path, data, file_type=SAVE_FILE_EXT)
