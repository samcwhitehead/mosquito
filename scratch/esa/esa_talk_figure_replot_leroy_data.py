"""
Script to re-plot data from LeRoy, et al., 2026 PLOS One
Want to more directly highlight mosquitoes as outliers

"""
# ======================================================
## IMPORTS
# ======================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

from mosquito.util import set_plot_params, my_adjust_axes

# ======================================================
## PARAMETERS
# ======================================================
# load environmental variables
load_dotenv()
DATA_ROOT = os.getenv("SAVE_DATA_PATH")

# path to LeRoy data
DATA_PATH = Path(DATA_ROOT) / "leroy_et_al_2026_data"
KINEMATICS_FILENAME = "Diptera_Kinematics_Aerodynamics.csv"
MORPHOLOGY_FILENAME = "Diptera_Morphology.csv"

# if and where to save
SAVE_FLAG = True
SAVE_PATH = "/media/sam/SamData/Dropbox/Conferences/ESA 2026/talk/figs"
SAVE_FILENAME = "leroy_2026_wbf_amp_plot"
SAVE_EXT = ".png"
SAVE_DPI = 500
ALL_GRAY_FLAG = False

if ALL_GRAY_FLAG:
    # color by family key
    infraorder_color_dict = {
        "Culicomorpha": '#555555',  # '#0077BB',  # '#004488',
        "Tipulomorpha": '#555555',  # cyan
    }
    SAVE_FILENAME += '_all_gray'
else:
    infraorder_color_dict = {
        "Culicomorpha": '#004488',  # '#0077BB',  # '#004488',
        "Tipulomorpha": '#555555',  # cyan
    }
default_color = '#555555'  # '#000000'  #  '#BBBBBB'
COLOR_MAPPING = defaultdict(lambda: default_color, infraorder_color_dict)
# Map the column to create a new color column: df["Color"] = df["Category"].map(color_mapping)
# Plot with Matplotlib: plt.bar(df["Category"], df["Values"], color=df["Color"])

# figure size options
FIGSIZE = (5, 4.25)
PLOT_TYPE = 'powerpoint'  # 'paper' or 'powerpoint' or 'poster


# ======================================================
## HELPER FUNCTIONS
# ======================================================
def load_leroy_data(data_path=DATA_PATH,
                    kinematics_filename=KINEMATICS_FILENAME,
                    morphology_filename=MORPHOLOGY_FILENAME):
    """
    Convenience function to load the data from LeRoy, et al. 2026 PLOS One paper

    Note that we want to basically just load kinematics data but add columns for
    the taxonomy

    Args:
        data_path: path to data file
        kinematics_filename: name of csv file containing kinematics data
        morphology_filename: name of csv file containing morphology data

    :return: leroy_df: pandas DataFrame containing paper data
    """
    # load the two csv files
    kinematics_df = pd.read_csv(os.path.join(data_path, kinematics_filename))
    morphology_df = pd.read_csv(os.path.join(data_path, morphology_filename))

    # for each row in the kinematics data frame, get the taxonomy info
    suborders = []
    infraorders = []
    families = []

    for idx, row in kinematics_df.iterrows():
        genus_species = row['genus_species']

        if isinstance(genus_species, str):
            morphology_idx = (morphology_df['genus_species'] == genus_species)

            suborders.append(morphology_df.loc[morphology_idx]['suborder'].values[0])
            infraorders.append(morphology_df.loc[morphology_idx]['infraorder'].values[0])
            families.append(morphology_df.loc[morphology_idx]['family'].values[0])
        else:
            suborders.append('')
            infraorders.append('')
            families.append('')

    # add taxonomy colonies to kinematics/aerodynamics data frame
    kinematics_df['suborder'] = suborders
    kinematics_df['infraorder'] = infraorders
    kinematics_df['family'] = families

    return kinematics_df


# -------------------------------------------------------------------------------
def plot_leroy_data(df,
                    color_map=COLOR_MAPPING,
                    figsize=FIGSIZE):
    """
    Function to make an updated plot with data from the Le Roy, et al., 2026 PLOS One paper
    on comparative dipteran flight

    :param df: pandas dataframe containing amp, wbf, and taxonomical info

    :return: fig, ax
    """
    # make figure and axis objects
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # convert data to numeric
    wbf = df['Wingbeat frequency'].apply(pd.to_numeric, errors='coerce')
    amp = df['Wingstroke amplitude'].apply(pd.to_numeric, errors='coerce')

    # rows of the data that are not NaN
    mask = amp.notna()

    # plot data
    ax.scatter(wbf[mask],
               amp[mask],
               c=df['infraorder'][mask].map(color_map),
               alpha=0.65,
               edgecolors='none',
               s=60,
               # edgecolors='black',
               # linewidths=0.5,
               )

    plt.tight_layout()

    # adjust axes
    ax = my_adjust_axes(ax, keep_spines=['left', 'bottom']) # , xticks=[0, 1000]) # , yticks=[0, 180])

    # label axes
    ax.set_xlabel('wingbeat frequency (Hz)')
    ax.set_ylabel('stroke amplitude (deg)')

    # set units for amplitude
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))

    plt.tight_layout()
    plt.show()
    return fig, ax

# # ======================================================
# ## MAIN FUNCTION
# # ======================================================
# def make_leroy_figure():
#     pass

# ======================================================
## MAIN SCRIPT
# ======================================================
if __name__ == "__main__":
    # plot params
    plot_params = set_plot_params(plot_type=PLOT_TYPE)

    # load data
    df = load_leroy_data()

    # make plot
    fig, ax = plot_leroy_data(df)

    # save?
    if SAVE_FLAG:
        fig.savefig(os.path.join(SAVE_PATH, SAVE_FILENAME + SAVE_EXT),
                    dpi=SAVE_DPI, bbox_inches='tight')