"""
Script to generate figures for my ESA talk showing
b1 steering muscle activity analysis

"""
# ======================================================
## IMPORTS
# ======================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from mosquito.process_abf import load_processed_data, detrend_emg
from mosquito.util import set_plot_params, my_adjust_axes, my_add_scalebar, add_radial_scalebar

from esa_talk_figures_burst_script import plot_data_trace, make_trace_figure, make_trace_figure_with_spike_times

# ======================================================
## PLOT PARAMETERS
# ======================================================
PLOT_TYPE = 'powerpoint'  # 'paper' or 'powerpoint' or 'poster
DARK_FLAG = False

# visualize plots as they're made?
VIZ_FLAG = False

# SAVE INFO
# should we save plots?
SAVE_EMG_FLAG = True  # if True, save the example emg trace plot; if False, don't
SAVE_MIC_FLAG = True   # if True, save putative MN activity plot; if False, don't
SAVE_HIST_FLAG = True  #  if True, save spike phase histogram plot; if False, don't

# how should figures be saved? (path, resolution, file format, etc)
SAVE_PATH = '/media/sam/SamData/Dropbox/Conferences/ESA 2026/talk/figs'
TRACE_SAVE_STR = 'steering_trace'  # header for filenames of trace plots
INSET_SAVE_STR = 'steering_inset'  # header for filenames of trace plots
MIC_INSET_SAVE_STR = 'mic_inset'
HIST_SAVE_STR = 'phase_hist'
SAVE_DPI = 500
SAVE_EXT = '.png'  #  '.svg'  #
SAVE_EXT_MN = '.png'  #  '.svg'  #

# params for trace plots
SCALEBAR_SIZE_SECONDS = 0.1  # 0.1
SCALEBAR_UNITS = 'ms'
SCALEBAR_SIZE_INSET = 0.05

FIGSIZE = (8.5, 1.25)  # (8, 2)
FIGSIZE_HIST = (2.5, 2.5)

# marker size and line width
MARKER_SIZE = 4

# spike phase histogram params
N_BINS = 32  # number of bins for phase histograms
HIST_BOTTOM = 0.3  # minimum radius for histogram
R_MAX = 2 # radius axis limit for plot

ALPHA = 0.65  # histogram face alpha

# dark plot options
if DARK_FLAG:
    TRACE_SAVE_STR += '_dark'
    plt.style.use('dark_background')

    # define some colors
    mosquito_female_color = '#ffbb77'
    mosquito_male_color = '#44aa99'
    drosophila_female_color = '#aaaaaa'  # '#DDAA33'
    albo_female_color = '#AA4499'  # should probably switch this to a dark version
else:
    plt.style.use('default')

    # define some colors
    mosquito_female_color = '#004488'
    mosquito_male_color = '#BB5566'
    drosophila_female_color = '#555555'  # '#DDAA33'
    albo_female_color = '#AA4499'

# trace time duration
TIME_DURATION = 0.75  # sec
INSET_TIME_DURATION = 0.30

# ======================================================
## EXAMPLE FLIES
# ======================================================
female_aegypti_b1 = {
    'folder': 84,
    'axo': 4,
    'channel': 0,
    't_range': (1.1, 1.1 + TIME_DURATION),
    'ylim': (-0.2, 0.25),
    'sex': 'F',
    'muscle': 'b1',
    'species': 'aegypti',
    'inset1_t_range': (1.1, 1.1 + INSET_TIME_DURATION),
    'color': mosquito_female_color,
}

female_melanogaster_b1 = {
    'folder': 82,
    'axo': 5,
    'channel': 0,
    't_range': (0.5, 0.5 + TIME_DURATION),
    'ylim': (-0.35, 0.45),
    'sex': 'F',
    'muscle': 'b1',
    'species': 'aegypti',
    'inset1_t_range': (3.65, 3.65 + INSET_TIME_DURATION),
    # 'inset1_t_range': (0.5, 0.5 + INSET_TIME_DURATION),
    'color': drosophila_female_color,
}

# ======================================================
## PLOT FUNCTIONS
# ======================================================
def plot_spike_phase_polar_hist(data,
                                channel=0,
                                ax=None,
                                n_bins=N_BINS,
                                bottom=HIST_BOTTOM,
                                r_max=R_MAX,
                                color='black',
                                alpha=ALPHA,
                                viz_flag=VIZ_FLAG):
    """
    Function to plot circular histogram of b1 spike phase

    Args:
        data: dictionary containing processed ephys data
        channel: which channel to draw from

    :return:
    """
    # read fields from data
    mic_phase = data['mic_phase']
    spike_idx = data['spike_idx']

    if isinstance(spike_idx, list):
        spike_idx = spike_idx[channel]

    # get histrogram counts
    counts, bin_edges = np.histogram(mic_phase[spike_idx], bins=n_bins, range=(-np.pi, np.pi), density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    width = 2 * np.pi / n_bins

    # make plot
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    ax.bar(bin_centers, counts, width=width, bottom=bottom, color=color, alpha=alpha)

    # plot parameters
    ax.set_rlim([0, r_max])
    # add_radial_scalebar(ax, angle_deg=180, cap_size=0, r_start=bottom)
    ax.set_rticks([])
    ax.set_xticks([0, np.pi])
    ax.set_xticklabels([0, r'$\pi$'])
    ax.grid(False)

    plt.tight_layout()

    # return
    if viz_flag:
        plt.show()

    return ax


# --------------------------------------------------------------
def make_polar_hist_figure(example_dict,
                           figsize=FIGSIZE_HIST,
                           save_flag=SAVE_HIST_FLAG,
                           save_path=SAVE_PATH,
                           save_ext=SAVE_EXT,
                           save_str=HIST_SAVE_STR,
                           save_dpi=SAVE_DPI,
                           **kwargs):
    """
    Convenience function to make spike phase polar histogram plots

    :param example_dict:
    :param save_flag:
    :param save_path:
    :param save_ext:
    :param save_str:
    :param kwargs:
    :return:
    """
    # load data
    data = load_processed_data(example_dict['folder'],
                               example_dict['axo'],
                               data_suffix='_spikes')

    # make plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    ax = plot_spike_phase_polar_hist(data, ax=ax, **kwargs)

    # save?
    if save_flag:
        save_filename = (f'{example_dict['species']}_{example_dict['sex']}_{example_dict['folder']}' +
                         f'_{example_dict['axo']:02d}_{save_str}{save_ext}')

        # save
        fig.savefig(os.path.join(save_path, save_filename), dpi=save_dpi, bbox_inches='tight')

    # return
    return fig, ax

# ======================================================
## RUN SCRIPT
# ======================================================
if __name__ == "__main__":
    # plot parameters
    plot_params = set_plot_params(plot_type=PLOT_TYPE)

    # which flies to plot for
    example_dict_list = [
        female_aegypti_b1,
        female_melanogaster_b1
    ]

    # loop through and make trace plots and burst plots
    for example_dict in example_dict_list:
        # EMG plots
        make_trace_figure(example_dict,
                          data_key='emg_filt',
                          scalebar_size=SCALEBAR_SIZE_SECONDS,
                          scalebar_time_units=SCALEBAR_UNITS,
                          figsize=FIGSIZE,
                          save_path=SAVE_PATH,
                          save_flag=SAVE_EMG_FLAG,
                          save_str=TRACE_SAVE_STR ,
                          save_ext=SAVE_EXT,
                          viz_flag=VIZ_FLAG,
                          )

        # spike phase histogram
        make_polar_hist_figure(example_dict,
                              figsize=FIGSIZE_HIST,
                              save_path=SAVE_PATH,
                              save_flag=SAVE_HIST_FLAG,
                              save_str=HIST_SAVE_STR ,
                              save_ext=SAVE_EXT,
                              viz_flag=VIZ_FLAG,
                              color=example_dict.get('color'))

        # make inset plots
        if 'inset1_t_range' in example_dict:
            # EMG plots
            make_trace_figure(example_dict,
                              time_range=example_dict['inset1_t_range'],
                              data_key='emg_filt',
                              scalebar_size=SCALEBAR_SIZE_INSET,
                              scalebar_time_units=SCALEBAR_UNITS,
                              figsize=FIGSIZE,
                              save_path=SAVE_PATH,
                              save_flag=SAVE_EMG_FLAG,
                              save_str=INSET_SAVE_STR,
                              save_ext=SAVE_EXT,
                              viz_flag=VIZ_FLAG,
                              )

            # MIC plots
            make_trace_figure_with_spike_times(example_dict,
                                              time_range=example_dict['inset1_t_range'],
                                              data_key='mic_filt',
                                              scalebar_size=SCALEBAR_SIZE_INSET,
                                              scalebar_time_units=SCALEBAR_UNITS,
                                              figsize=FIGSIZE,
                                              line_color='black',
                                               linewidth=0.5,
                                               marker_size=MARKER_SIZE,
                                              marker_color=example_dict['color'],
                                              ylim=(-0.2,1.3),
                                              norm_flag=True,
                                              save_path=SAVE_PATH,
                                              save_flag=SAVE_MIC_FLAG,
                                               save_str=MIC_INSET_SAVE_STR,
                                               save_ext=SAVE_EXT,
                                              viz_flag=VIZ_FLAG)