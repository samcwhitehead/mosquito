"""
Temp code to look at b1 vs b2 spikes

"""
# Imports
import os
import glob
import pickle
import pywt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal

from mosquito.process_abf import load_processed_data

# ---------------------------------------
# MAIN
# ---------------------------------------
# Run script
if __name__ == "__main__":
    # loading separate data for b1 and b2
    data_folder1 = 50
    axo_num1 = 32
    data_folder2 = 50  # 46
    axo_num2 = 31  # 6

    # get data files
    try:
        data1 = load_processed_data(data_folder1, axo_num1, data_suffix='_spikes')
    except ValueError:
        data1 = load_processed_data(data_folder1, axo_num1)

    try:
        data2 = load_processed_data(data_folder2, axo_num2, data_suffix='_spikes')
    except ValueError:
        data2 = load_processed_data(data_folder2, axo_num2)

    data_list = [data1, data2]
    data_labels = ['b1', 'b2']

    # plot emg and spike detection
    fig, ax_list = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_list.ravel()

    tmin = 0
    tmax = 100

    for ith, data in enumerate(data_list):
        t = data['time']
        emg_filt = data['emg_filt']
        emg = data['emg']
        try:
            good_spike_idx = data['good_spike_idx']
        except KeyError:
            good_spike_idx = data['spike_idx']

        mask = (t >= tmin) & (t <= tmax)
        mask_spikes = (t[good_spike_idx] >= tmin) & (t[good_spike_idx] <= tmax)

        ax_list[ith].plot(t[mask], emg_filt[mask])
        ax_list[ith].plot(t[good_spike_idx][mask_spikes], emg_filt[good_spike_idx][mask_spikes], 'rx')


    fig.tight_layout()
    plt.show()
