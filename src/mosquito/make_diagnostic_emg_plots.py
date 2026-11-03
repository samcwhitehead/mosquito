"""
Quick and dirty script to make plots for a series of experiment folder
that will let me quickly assess the quality of recordings

"""
# ---------------------------------
## IMPORTS
# ---------------------------------
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pathlib import Path

from mosquito.process_abf import load_processed_data
from mosquito.analyze_bursts import run_spike_detection

# ---------------------------------
## PARAMS
# ---------------------------------
# load environmental variables
load_dotenv()

# set path information
DATA_ROOT = os.getenv('SAVE_DATA_PATH')
PLOT_SAVE_PATH = Path(DATA_ROOT) / 'albo_diagnostic_plots'
PLOT_SAVE_PATH.mkdir(parents=False, exist_ok=True)

# which data folders to analyze
EXPR_NUMBERS = [95, 97, 104, 113, 114, 115, 116, 117, 118]


# ---------------------------------
## HELPER FUNCTIONS
# ---------------------------------
def make_emg_plot(data):
    """
    Quick helper function to make a plot of emg
    :param data: dictionary from processed pickle file
    :return: fig, ax (figure and axis handles to plot)
    """
    # initialize axis and figure
    fig, ax = plt.subplots(figsize=(11, 4))

    # read out variables
    emg = data['emg'][0]
    spike_idx = data['spike_idx']
    fs = data['sampling_freq']

    t = (1 / fs) * np.arange(emg.size)

    # plot data
    ax.plot(t, emg)
    ax.plot(t[spike_idx], emg[spike_idx], 'rx')

    # label axes
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_title('EMG')
    ax.set_ylabel('emg (V)')
    ax.set_xlabel('time (s)')

    plt.show()

    return fig, ax


def make_waveform_plot(data):
    run_spike_detection(data, viz_flag=True)



# ---------------------------------
## RUN MAIN SCRIPT
# ---------------------------------
if __name__ == "__main__":
    # loop over data folders
    for expr_num in EXPR_NUMBERS:
        # get the actual folder for this experiment number
        pattern = f"{int(expr_num)}_????????"
        matches = list(Path(DATA_ROOT).glob(pattern))
        matches = [
            p for p in Path(DATA_ROOT).glob(pattern)
            if p.is_dir()
        ]

        if not matches:
            raise FileNotFoundError(f"No folder found for {expr_num}")

        if len(matches) > 1:
            raise ValueError(f"Multiple folders found for {expr_num}: {matches}")

        data_folder = matches[0]

        # find all axo (.abf) files in that given folder
        abf_files = list(Path(data_folder).glob('**/*_processed.pkl'))

        # loop over each file
        for file in sorted(abf_files):
            # laod data
            print(file)
            data = pickle.load(open(file, "rb"))
            data['filepath_load'] = file  # add the full filepath that this was just loaded from

            # plot emg
            make_emg_plot(data)
