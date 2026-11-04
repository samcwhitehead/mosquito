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

from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict


# ======================================================
## PLOT PARAMETERS
# ======================================================
# load environmental variables
load_dotenv()
DATA_ROOT = os.getenv("SAVE_DATA_PATH")

# path to LeRoy data
DATA_PATH = Path(DATA_ROOT) / "leroy_et_al_2026_data"
KINEMATICS_FILENAME = "Diptera_Kinematics_Aerodynamics.csv"
MORPHOLOGY_FILENAME = "Diptera_Morpholog.csv"

# color by family key
family_color_dict = {
    "Culicidae": '#004488',
    "Tipulidae": '33BBEE',  # cyan
}
default_color = '#555555'
color_mapping = defaultdict(lambda: default_color, family_color_dict)
# Map the column to create a new color column: df["Color"] = df["Category"].map(color_mapping)
# Plot with Matplotlib: plt.bar(df["Category"], df["Values"], color=df["Color"])

# figure size options
FIGSIZE = (5, 5)



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

    for row in kinematics_df.iterrows():
        genus_species = row['genus_species']

        morphology_idx = (morphology_df['species'] == genus_species)

        suborders.append(morphology_df.loc[morphology_idx]['suborder'])
        infraorders.append(morphology_df.loc[morphology_idx]['infraorder'])
        families.append(morphology_df.loc[morphology_idx]['family'])

    # add taxonomy colonies to kinematics/aerodynamics data frame
    kinematics_df['suborder'] = suborders
    kinematics_df['infraorder'] = infraorders
    kinematics_df['family'] = families

    return kinematics_df


# ======================================================
## MAIN FUNCTION
# ======================================================
def make_leroy_plot(df):
    """
    Function to make an updated plot with data from the Le Roy, et al., 2026 PLOS One paper
    on comparative dipteran flight

    :param df: pandas dataframe containing amp, wbf, and taxonomical info

    :return: fig, ax
    """


# ======================================================
## MAIN SCRIPT
# ======================================================
if __name__ == "__main__":
    pass