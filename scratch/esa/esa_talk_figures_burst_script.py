"""
Code to make power muscle recording figures for my ESA 2026 talk


"""
# ======================================================
## IMPORTS
# ======================================================
import os

import numpy as np
import matplotlib.pyplot as plt

from mosquito.process_abf import load_processed_data, detrend_emg
from mosquito.util import set_plot_params, my_adjust_axes, my_add_scalebar
from mosquito.analyze_bursts import run_spike_detection, load_burst_data

# ======================================================
## PLOT PARAMETERS
# ======================================================
PLOT_TYPE = 'powerpoint'  # 'paper' or 'powerpoint' or 'poster
DARK_FLAG = False

# visualize plots as they're made?
VIZ_FLAG = True

# params for trace plots
SCALEBAR_SIZE = 0.5  # 0.1
SCALEBAR_UNITS = 's'
SCALEBAR_SIZE_INSET = 0.02
SCALEBAR_LINEWIDTH = 2.0

FIGSIZE = (6, 2)# (8, 2)
FIGSIZE_INSET = (1.75, 2)# (8, 2)
FIGSIZE_INSET_MN = (1.75, 0.75)# (8, 2)
# FIGSIZE_SPIKE_NUM = (3.65, 4)

LINEWIDTH = 0.75

MARKERSIZE=5
MARKERTYPE='o'
MARKERCOLOR='black'

# SAVE INFO
# should we save plots?
SAVE_TRACE_FLAG = True  # if True, save the example emg trace plot; if False, don't
SAVE_MN_FLAG = True  # if True, save putative MN activity plot; if False, don't

# how should figures be saved? (path, resolution, file format, etc)
SAVE_PATH = '/media/sam/SamData/Dropbox/Conferences/ESA 2026/talk/figs'
TRACE_SAVE_STR = 'trace'  # header for filenames of trace plots
MN_SAVE_STR = 'mn'  # header for filenames of trace plots
SAVE_DPI = 500
SAVE_EXT_TRACE = '.png'  #  '.svg'  #
SAVE_EXT_MN = '.png'  #  '.svg'  #

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
TIME_DURATION = 5  # sec


# ======================================================
## EXAMPLE FLIES
# ======================================================
female_aegypti_dlm = {
    'folder': 26,
    'axo': 2,
    'channel': 0,
    't_range': (1.4, 1.4 + TIME_DURATION),
    'ylim': (-0.13, 1.4),
    'sex': 'F',
    'muscle': 'DLM',
    'species': 'aegypti',
    'color': mosquito_female_color,
}

female_aegypti_dvm = {
    'folder': 38,
    'axo': 10,
    'channel': 0,
    't_range': (98, 98 + + TIME_DURATION),
    'ylim':(-0.28, 2.1),
    'sex': 'F',
    'muscle': 'DVM',
    'species': 'aegypti',
    'color': mosquito_female_color,
    'inset1_t_range': (78.42, 78.42 + 0.05),
    'inset2_t_range': (79.843, 79.843 + 0.05),
}

female_drosophila_dlm = {
    'folder': 35,
    'axo': 0,
    'channel': 0,
    't_range': (0.9, 0.9 + + TIME_DURATION),
    'ylim':(-0.35, 4.75),
    'sex': 'F',
    'muscle': 'DLM',
    'species': 'melanogaster',
    'color': drosophila_female_color,
    'inset1_t_range': (78.50, 78.50+0.05),
    'inset2_t_range': (79.838, 79.838+0.05),
}

female_drosophila_dvm = {
    'folder': 34,
    'axo': 1,
    'channel': 0,
    't_range': (1, 1 + TIME_DURATION),
    'ylim':(-0.14, 1.05),
    'sex': 'F',
    'muscle': 'DVM',
    'species': 'melanogaster',
    'color': drosophila_female_color,
}

female_albo_dlm = {
    'folder': 104,
    'axo': 7,
    'channel': 0,
    't_range': (1, 1 + TIME_DURATION),
    'ylim': (-0.3, 3.5),
    'sex': 'F',
    'muscle': 'DLM',
    'species': 'albopictus',
    'inset1_t_range': (0.78, 0.78+0.08),
    'color': albo_female_color,
}

examples_list = [female_aegypti_dlm, female_aegypti_dvm, female_drosophila_dlm, female_drosophila_dvm, female_albo_dlm]

# ======================================================
## HELPER FUNCTIONS
# ======================================================
def plot_data_trace(data,
                    key,
                    time_range,
                    channel=0,
                    ax=None,
                    scalebar_size=SCALEBAR_SIZE,
                    ylim=None,
                    time_units=SCALEBAR_UNITS,
                    line_color='k',
                    linewidth=LINEWIDTH,
                    scalebar_linewidth=SCALEBAR_LINEWIDTH,
                    norm_flag=False):
    """
    Function to take in a data dictionary for a single fly and make
    a zoomed-out plot of the time series data trace in 'key'

    Args:
        data: dictionary loaded from pickled *_processed.p file
        key: string, giving data dict key for variable to plot
        channel: if plotting multichannel data, give channel number
        time_range: tuple giving the range of data to plot
            (e.g. (4, 8) will plot between t=4 and t=8 seconds)
        ax: axis object to plot to
        scalebar_size: size (duration) of scalebar, in seconds
        figsize: tuple giving (width, height) of figure in inches
            (passed to pyplot.subplots)
        ylim: tuple giving y axis limits. If None, use auto scale
        time_units: units of time values in data. should be 's'
        fig_params: dictionary containing plot params; under
            construction

    Returns:
        fig, ax
    """
    # read out data from dict
    t = data['time']
    dat = data[key]

    if isinstance(dat, list):
        dat = dat[channel]

    if key == 'emg':
        dat = detrend_emg(dat)

    # plot params
    color = line_color

    # initialize figure (if needed)
    if ax is None:
        fig, ax = plt.subplots()

    # restrict time domain to time_range
    mask = (t >= time_range[0]) & (t <= time_range[-1])

    dat_curr = dat[mask]
    t_curr = t[mask]

    if norm_flag:
        dat_curr = (dat_curr - dat_curr.min()) / (dat_curr.max() - dat_curr.min())

    # plot. Note that we're subtracting off the initial time so it always
    # starts at t=0
    ax.plot(t_curr - time_range[0],
            dat_curr,
            color=color,
            lw=linewidth)

    # -----------------------------------------
    # axis properties
    # restrict x range
    ax.set_xlim([0, time_range[1] - time_range[0]])

    # restrict y range?
    if ylim is not None:
        ax.set_ylim(ylim)

    # add scalebar
    if scalebar_size <= 0:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])

    else:
        ax.spines['bottom'].set_bounds([0, scalebar_size])
        ax.spines['bottom'].set_linewidth(scalebar_linewidth)
        ax.spines['bottom'].set_position(('outward', 3))

        # label scalebar.
        if time_units == 's':
            label_val = scalebar_size
        elif time_units == 'ms':
            label_val = int(1000*scalebar_size)
        else:
            print('Unrecognized time units')

        ax.set_xticks([scalebar_size / 2])
        ax.set_xticklabels([f'{label_val} {time_units}'])
        ax.tick_params(length=0)

    # remove other axis lines
    for spine in ['right', 'top', 'left']:
        ax.spines[spine].set_visible(False)

    # remove y ticks
    ax.set_yticks([])

    return ax


# -------------------------------------------------------------------------
def plot_putative_mn_spikes(ax,
                            data,
                            spike_idx_all,
                            time_range,
                            line_width=1,
                            line_color='k',
                            line_height=1,
                            scalebar_size=0):
    """
    Function to make a plot of inferred MN spikes corresponding to a given
    burst

    Args:
        - data: dictionary loaded from pickled *_processed.p file
        - spike_idx_all: numpy array giving indices of ALL spikes, i.e.
            the ones within a burst
        - time_range: tuple giving the range of data to plot
        - line_width: width of line
        - line_color: color of line
        - line_height: height of line
        - scalebar_size: size (duration) of scalebar, in seconds

    Returns:
        - ax: handle to axis of plot
    """
    # some params
    scalebar_linewidth = 3.5

    # first get indices corresponding to time ranges
    t = data['time']

    idx_all = np.arange(t.size)
    idx_range = idx_all[(t >= time_range[0]) & (t <= time_range[-1])]

    in_range_idx = (spike_idx_all >= idx_range[0]) & (spike_idx_all <= idx_range[-1])
    spike_idx = spike_idx_all[in_range_idx]
    # in_range_idx = (spikes_df['peak_idx_global'] >= idx_range[0]) & (spikes_df['peak_idx_global'] <= idx_range[-1])
    # spike_idx = spikes_df['peak_idx_global'].loc[in_range_idx]

    # plot putative spikes to axes
    for idx in spike_idx:
        ax.axvline(t[idx] - time_range[0], ymin=0.01, ymax=line_height,
                   color=line_color, linewidth=line_width)

    # also plot baseline
    ax.axhline(0, color=line_color, linewidth=2 * line_width)

    # adjust axes
    # add scalebar?
    if scalebar_size == 0:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])

    else:
        if scalebar_size < 1:
            units = 'ms'
            label_mult = 1000
        else:
            units = 's'
            label_mult = 1

        ax = my_add_scalebar(ax, (0, scalebar_size), linewidth=scalebar_linewidth,
                             spine='bottom', units=units, ticklength=0,
                             offset_length=10, label_mult_factor=label_mult)

    # remove other axis lines
    for spine in ['right', 'top', 'left']:
        ax.spines[spine].set_visible(False)

    # remove y ticks
    ax.set_yticks([])

    # set ylin
    ax.set_ylim([-0.00001, 1.1])
    ax.set_xlim([0, time_range[-1] - time_range[0]])

    return ax


# ----------------------------------------------------------------------------------------------------
def plot_spike_times_on_trace(data,
                              time_range,
                              ydata_key='mic',
                              channel=0,
                              ax=None,
                              marker_size=MARKERSIZE,
                              marker_color=MARKERCOLOR,
                              marker_type=MARKERTYPE,
                              norm_flag=False):
    """
    Function to add to an existing plot by plotting spike times atop data trace

    :return:
    """
    # get axis object if we don't have it
    if ax is None:
        ax = plt.gca()

    # turn off axis autoscale
    ax.set_autoscale_on(False)

    # read out data
    t = data['time']

    ydata = data[ydata_key]
    if isinstance(ydata, list):
        ydata = ydata[channel]

    # normalize signal to plot on top of
    if norm_flag:
        mask = (t >= time_range[0]) & (t <= time_range[-1])
        ydata = (ydata - ydata[mask].min()) / (ydata[mask].max() - ydata[mask].min())

    spike_idx = data['spike_idx']
    if isinstance(spike_idx, list):
        spike_idx = spike_idx[channel]

    # plot
    ax.plot(t[spike_idx] - time_range[0],
            ydata[spike_idx],
            color=marker_color,
            markersize=marker_size,
            marker=marker_type,
            linestyle='None')

    return ax


# ======================================================
## MAIN FUNCTIONS
# ======================================================
def make_trace_figure(example_dict,
                      time_range=None,
                      data_key='emg',
                      channel=None,
                      ylim=None,
                      line_color=None,
                      figsize=FIGSIZE,
                      scalebar_size=SCALEBAR_SIZE,
                      scalebar_time_units=SCALEBAR_UNITS,
                      norm_flag=False,
                      save_path=SAVE_PATH,
                      save_ext=SAVE_EXT_TRACE,
                      save_dpi=SAVE_DPI,
                      save_str=TRACE_SAVE_STR,
                      save_flag=SAVE_TRACE_FLAG,
                      viz_flag=True,
                      **kwargs):
    """
    Wrapper function to make a time series trace plot 
    and save it
    
    Args:
        example_dict: dictionary containing example data
            identifiers, including data folder, axo, 
            channel, t_range, ylim, sex, muscle, species,
            and color for the plot
        data_key: which type of data to plot ('emg', 
            'wbf', etc.)
        time_range: tuple giving time range to plot over.
            If None, read from example_dict
        figsize: tuple giving size of figure in inches
        scalebar_size: size of scalebar (in time units)
        save_path: path to save figure
        save_ext: file extension to save trace plot
        save_dpi: dpi for saving figure
        save_str: string to append to filename
        save_flag: whether to save trace plot
        viz_flag: bool. show plots as they're made?
    Returns: 
        fig: handle to trace figure
    
    """
    # load data
    data = load_processed_data(example_dict['folder'], example_dict['axo'])

    # define time range
    if time_range is None:
        time_range = example_dict['t_range']

    if ylim is None:
        ylim = example_dict['ylim']

    if channel is None:
        channel = example_dict['channel']

    if line_color is None:
        line_color = example_dict['color']


    # make figure
    fig, ax = plt.subplots(figsize=figsize)
    ax = plot_data_trace(data,
                         data_key,
                         time_range,
                         channel=channel,
                         ax=ax,
                         scalebar_size=scalebar_size,
                         time_units=scalebar_time_units,
                         ylim=ylim,
                         line_color=line_color,
                         norm_flag=norm_flag,
                         **kwargs)

    # # kludgy but whatever
    # if plot_spike_times_flag:
    #     ax = plot_spike_times_on_trace(data, time_range, data_key, ax=ax,
    #                                    channel=example_dict['channel'],
    #                                    marker_color=line_color)

    if save_flag:
        # make the filename of the figure reflect which data file it's from
        save_filename = (f'{example_dict['species']}_{example_dict['sex']}_{example_dict['folder']}' +
                         f'_{example_dict['axo']:02d}_{data_key}_{save_str}{save_ext}')

        # save
        fig.savefig(os.path.join(save_path, save_filename), dpi=save_dpi, bbox_inches='tight')

    plt.tight_layout()
    if viz_flag:
        plt.show()

    return fig


# ---------------------------------------------------------------------------------------
def make_trace_figure_with_spike_times(example_dict,
                                       time_range=None,
                                       data_key='emg',
                                       channel=None,
                                       ylim=None,
                                       line_color=None,
                                       marker_color=None,
                                       marker_size=MARKERSIZE,
                                       figsize=FIGSIZE,
                                       scalebar_size=SCALEBAR_SIZE,
                                       scalebar_time_units=SCALEBAR_UNITS,
                                       norm_flag=False,
                                       save_path=SAVE_PATH,
                                       save_ext=SAVE_EXT_TRACE,
                                       save_dpi=SAVE_DPI,
                                       save_str=TRACE_SAVE_STR,
                                       save_flag=SAVE_TRACE_FLAG,
                                       viz_flag=True,
                                       **kwargs):
    """
    Wrapper function to make a time series trace plot
    and save it

    Args:
        example_dict: dictionary containing example data
            identifiers, including data folder, axo,
            channel, t_range, ylim, sex, muscle, species,
            and color for the plot
        data_key: which type of data to plot ('emg',
            'wbf', etc.)
        time_range: tuple giving time range to plot over.
            If None, read from example_dict
        figsize: tuple giving size of figure in inches
        scalebar_size: size of scalebar (in time units)
        save_path: path to save figure
        save_ext: file extension to save trace plot
        save_dpi: dpi for saving figure
        save_str: string to append to filename
        save_flag: whether to save trace plot
        viz_flag: bool. show plots as they're made?
    Returns:
        fig: handle to trace figure

    """
    # load data
    data = load_processed_data(example_dict['folder'], example_dict['axo'])

    # define time range
    if time_range is None:
        time_range = example_dict['t_range']

    if ylim is None:
        ylim = example_dict['ylim']

    if channel is None:
        channel = example_dict['channel']

    if line_color is None:
        line_color = example_dict['color']

    # make figure
    fig, ax = plt.subplots(figsize=figsize)
    ax = plot_data_trace(data,
                         data_key,
                         time_range,
                         channel=channel,
                         ax=ax,
                         scalebar_size=scalebar_size,
                         time_units=scalebar_time_units,
                         ylim=ylim,
                         line_color=line_color,
                         norm_flag=norm_flag,
                         **kwargs)


    ax = plot_spike_times_on_trace(data, time_range, data_key, ax=ax,
                                   channel=example_dict['channel'],
                                   marker_color=marker_color,
                                   norm_flag=norm_flag,
                                   marker_size=marker_size)

    if save_flag:
        # make the filename of the figure reflect which data file it's from
        save_filename = (f'{example_dict['species']}_{example_dict['sex']}_{example_dict['folder']}' +
                         f'_{example_dict['axo']:02d}_{data_key}_{save_str}{save_ext}')

        # save
        fig.savefig(os.path.join(save_path, save_filename), dpi=save_dpi, bbox_inches='tight')

    plt.tight_layout()
    if viz_flag:
        plt.show()

    return fig


# ---------------------------------------------------------------------------------------
def make_putative_mn_spike_figure(example_dict,
                                  spike_idx_all,
                                  time_range=None,
                                  figsize=FIGSIZE_INSET_MN,
                                  save_path=SAVE_PATH,
                                  save_ext=SAVE_EXT_MN,
                                  save_dpi=SAVE_DPI,
                                  save_str=MN_SAVE_STR,
                                  save_flag=SAVE_MN_FLAG,
                                  viz_flag=True,
                                  **kwargs):
    """

    Args:
        example_dict: dictionary containing example data
        spike_idx_all: array of index of ALL spike times, i.e. within
            burst spikes
        time_range: tuple giving time range to plot over.
            If None, read from example_dict
        figsize: tuple giving size of figure in inches
        scalebar_size: size of scalebar (in time units)
        save_path: path to save figure
        save_ext: file extension to save trace plot
        save_dpi: dpi for saving figure
        save_str: string to append to filename
        save_flag: whether to save trace plot
        viz_flag: bool. show plots as they're made?
        kwargs: arguments to pass to plot_putative_mn_spikes

    Returns:
        ax: handle to mn activity plot
    """
    # load data
    data = load_processed_data(example_dict['folder'], example_dict['axo'])

    # define time range
    if time_range is None:
        time_range = example_dict['t_range']

    # make fig and axis handles
    fig, ax = plt.subplots(figsize=figsize)

    # plot
    ax = plot_putative_mn_spikes(ax,
                                 data,
                                 spike_idx_all,
                                 time_range,
                                 **kwargs)

    if save_flag:
        # make the filename of the figure reflect which data file it's from
        save_filename = (f'{example_dict['species']}_{example_dict['sex']}_{example_dict['folder']}' +
                         f'_{example_dict['axo']:02d}_{data_key}_{save_str}{save_ext}')

        # save
        fig.savefig(os.path.join(save_path, save_filename), dpi=save_dpi, bbox_inches='tight')

    plt.tight_layout()
    if viz_flag:
        plt.show()

    return fig


# ======================================================
## RUN SCRIPT
# ======================================================
if __name__ == "__main__":
    # plot parameters
    plot_params = set_plot_params(plot_type=PLOT_TYPE)

    # which flies to plot for
    example_dict_list = [female_albo_dlm]

    # which type of plot to make
    data_key = 'emg'

    # loop through and make trace plots and burst plots
    for example_dict in example_dict_list:
        # trace plots
        make_trace_figure(example_dict,
                          data_key=data_key,
                          figsize=FIGSIZE,
                          save_path=SAVE_PATH,
                          save_flag=SAVE_TRACE_FLAG,
                          viz_flag=VIZ_FLAG)

        # make inset plots
        for t_range_string in ['inset1_t_range', 'inset2_t_range']:

            if t_range_string in example_dict:
                # get string to save with
                save_str_curr = t_range_string.split('_')[0]

                # make plot of ephys waveform
                make_trace_figure(example_dict,
                                  time_range=example_dict[t_range_string],
                                  data_key=data_key,
                                  figsize=FIGSIZE_INSET,
                                  scalebar_size=0,
                                  save_path=SAVE_PATH,
                                  save_str=save_str_curr,
                                  save_flag=SAVE_TRACE_FLAG,
                                  viz_flag=VIZ_FLAG)

                # get per-burst spike information to plot putative MN activity
                if example_dict['species'] == 'melanogaster':
                    data_temp = load_processed_data(example_dict['folder'], example_dict['axo'])
                    spike_idx_all = data_temp['spike_idx']
                else:
                    try:
                        spike_df = load_burst_data(example_dict['folder'], example_dict['axo'])
                        spike_idx_all = spike_df['peak_idx_global'].values
                    except ValueError:
                        print(
                            f"No intra-burst spike data for {example_dict['folder']}, axo {example_dict['axo']}."
                             " Run run_spike_detection from analyze_bursts.py first and save results"
                        )


                # make plot of putative MN activity
                make_putative_mn_spike_figure(example_dict,
                                              spike_idx_all,
                                              time_range=example_dict[t_range_string],
                                              figsize=FIGSIZE_INSET_MN,
                                              save_path=SAVE_PATH,
                                              save_ext=SAVE_EXT_MN,
                                              save_dpi=SAVE_DPI,
                                              save_str='_'.join([save_str_curr, MN_SAVE_STR]),
                                              save_flag=SAVE_MN_FLAG,
                                              viz_flag=VIZ_FLAG)




