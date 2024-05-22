"""
Some code to make transferring data between machines a little easier

TODO:
    - add README copying?

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
import pyabf
import shutil

import numpy as np

from pathlib import Path
from datetime import date, datetime


# ---------------------------------------
# PARAMS
# ---------------------------------------
# date for the data we want to transfer right now. if None, ask for input
# can be in some isoform (e.g. 'YYYY-MM-DD') or date.today()
DATE = None

# define paths to data
root_path = '/media/sam/SamData/Mosquitoes/testing'
vid_path = os.path.join(root_path, 'vid_data')  # folder with high speed video
axo_path = os.path.join(root_path, 'axo_data')  # folder with abf files
out_path = os.path.join(root_path, 'out')   # folder to save output to


# ---------------------------------------
# FUNCTIONS
# ---------------------------------------
# -------------------------------------------------------------------------
def vid_filename_to_tstamp(fn):
    """
    Quick function to grab the date/time from a highspeed video file
    and convert it to a timestamp
    """
    ymd = fn.split('_')[-2]
    hms = fn.split('_')[-1]
    date_str = '{}T{}'.format(ymd, hms)
    date_time = datetime.fromisoformat(date_str)

    return date_time.timestamp()


# -------------------------------------------------------------------------
def get_next_folder_name(folder_date, out_path=out_path):
    """
    Quick function to get the name for the next folder we should make
    in form XX_YYYYMMDD, where XX is an index

    Args:
        folder_date: date to use for folder
        out_path: directory where stuff is saved

    Returns: next_folder, string name of next folder
    """
    # get index for next folder
    curr_folders = [f for f in os.listdir(out_path) if os.path.isdir(os.path.join(out_path, f))]
    curr_folder_ind_strs = [f.split('_')[0] for f in curr_folders if f[0].isdigit() and f[1].isdigit()]

    if len(curr_folder_ind_strs) < 1:
        next_folder_ind = 0
    else:
        curr_folder_inds = [int(f) for f in curr_folder_ind_strs]
        next_folder_ind = max(curr_folder_inds) + 1

    # put in date info
    next_folder = '{:02d}_'.format(next_folder_ind) + folder_date.strftime("%Y%m%d")

    return next_folder


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    # ----------------------------------------------
    # set date for current folder
    if DATE is None:
        date_user = input("Enter date for data to transfer in form YYYY-MM-DD:\n")
        folder_date = date.fromisoformat(date_user)
    else:
        folder_date = date.fromisoformat(DATE)   # use folder_date = date.today() if doing folder for same day

    print('Fetching data for {}'.format(folder_date))

    # ----------------------------------------------
    # get next folder name
    next_folder = get_next_folder_name(folder_date, out_path=out_path)

    # ----------------------------------------------
    # get axo files that match folder_date
    axo_dir = glob.glob(os.path.join(axo_path, '*.abf'))  # [fn for fn in os.listdir(axo_path) if fn.endswith('.abf')]

    # take only ones that belong to current date
    axo_dir_fns = [os.path.basename(fn) for fn in axo_dir]
    axo_dates = ['-'.join(fn.split('_')[:-1]) for fn in axo_dir_fns]
    axo_dates_match_idx = [(date.fromisoformat(ad) == folder_date) for ad in axo_dates]
    axo_dir_match = [axo_dir[ith] for ith in range(len(axo_dir)) if axo_dates_match_idx[ith]]
    axo_dir_match = sorted(axo_dir_match)

    # ----------------------------------------------
    # get info associated with these files
    axo_creation_times = [os.path.getctime(adm) for adm in axo_dir_match]
    # axo_stems = [Path(fn).stem for fn in axo_dir_match]
    # axo_numbers = [int(fn.split('_')[-1]) for fn in axo_stems]

    # ----------------------------------------------
    # get video file info
    vid_dir = glob.glob(os.path.join(vid_path, '*/*.mraw'))
    vid_fns = [os.path.splitext(os.path.basename(vd))[0] for vd in vid_dir]

    # get timestamps for videos
    vid_tstamps = [vid_filename_to_tstamp(fn) for fn in vid_fns]

    # ----------------------------------------------
    # get indices for matching vids to axo
    axo_bins = axo_creation_times.copy()
    final_abf = pyabf.ABF(axo_dir_match[-1])
    final_abf_duration = final_abf.dataLengthSec
    axo_bins.append(final_abf_duration + axo_bins[-1])

    vid_abf_ind = np.digitize(np.asarray(vid_tstamps), bins=axo_bins) - 1

    # ----------------------------------------------
    # copy things over
    # make directory for current experiment
    expr_folder = os.path.join(out_path, next_folder)
    if not os.path.exists(expr_folder):
        os.mkdir(expr_folder)

    # loop over axo files and add to their own folders
    for ith, axo_fn in enumerate(axo_dir_match):
        # create folder for current axo file
        curr_folder = os.path.join(expr_folder, Path(axo_fn).stem)
        if not os.path.exists(curr_folder):
            os.mkdir(curr_folder)

        # copy axo file to it
        dst_path = os.path.join(curr_folder, os.path.split(axo_fn)[-1])
        if not os.path.exists(dst_path):
            print('Copying {} \n to {} ...'.format(axo_fn, dst_path))
            shutil.copy2(axo_fn, dst_path)

        # if there are video files associated with this axo file, copy those as well
        curr_vid_ind = np.where(vid_abf_ind == ith)[0]
        for cvi in curr_vid_ind:
            vid_src_path = vid_dir[cvi]
            vid_dst_folder = os.path.join(curr_folder, vid_fns[cvi])
            if not os.path.exists(vid_dst_folder):
                os.mkdir(vid_dst_folder)

            vid_dst_path = os.path.join(vid_dst_folder, vid_fns[cvi])
            if not os.path.exists(vid_dst_path):
                print('Copying {} \n to {} ...'.format(vid_src_path, vid_dst_path))
                shutil.copy2(vid_src_path, vid_dst_path)

    print('=========== \n Done \n ===========')
