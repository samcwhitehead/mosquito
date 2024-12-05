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
import imageio.v3 as iio

from pathlib import Path
from datetime import date, datetime


# ---------------------------------------
# PARAMS
# ---------------------------------------
# date for the data we want to transfer right now. if None, ask for input
# can be in some isoform (e.g. 'YYYY-MM-DD') or date.today()
DATE = None

# # define paths to data
# vid_path = os.path.normpath(
#     'C:\\Users\\swhitehe\\Documents\\Photron\\PFV4\\mosquito_testing')  # folder with high speed video
# snapshot_path = os.path.normpath(
#     'C:\\Users\\swhitehe\\Documents\\Photron\\PFV4\snapshots')  # folder with high speed video
# axo_path = os.path.normpath('C:\\Users\\swhitehe\\Documents\\Molecular Devices\\pCLAMP\\Data')  # folder with abf files
# out_path_list = [os.path.normpath('E:\\HighSpeedVideo\\mosquito'),
#                  os.path.normpath('E:\\Mosquitoes'),
#                  ]  # folder to save output to

# define paths to data
root_path = '/media/sam/SamData/Mosquitoes/testing'
vid_path = os.path.join(root_path, 'vid_data')  # folder with high speed video
snapshot_path = os.path.join(root_path, 'snapshots')  # folder with snapshots of electrode position
axo_path = os.path.join(root_path, 'axo_data')  # folder with abf files
out_path = os.path.join(root_path, 'out')  # folder to save output to

# out_path = ''
# cc = 0
# while not os.path.exists(out_path):
#     out_path = out_path_list[cc]
#     cc += 1

# README file info
readme_rows = ['Date', 'Species', 'Eclosion date', 'Amplifier',
               'Odor stimulus type', 'Odor stimulus flow rate',
               'HPA flow rate', 'Notes']


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
def get_jpg_tstamp(jpg_path):
    """
    Quick function to grab the date/time from a jpg file
    """
    metadata = iio.immeta(jpg_path)
    exif = metadata['exif']
    exif_split = [x for x in exif.split(b'\x00') if not x == b'']
    exif_date = exif_split[-1].decode('utf-8')
    exif_ymd, exif_hms = exif_date.split(' ')
    exif_date_iso = f'{exif_ymd.replace(":", "-")}T{exif_hms}'
    date_time = datetime.fromisoformat(exif_date_iso)

    return date_time.timestamp()


# -------------------------------------------------------------------------
def get_abf_tstamp(abf_path):
    """
    Quick function to grab the date/time from an abf file
    """
    abf = pyabf.ABF(abf_path)
    date_time = abf.abfDateTime

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


# -------------------------------------------------------------------------
def deal_files(expr_folder, to_deal_dir, file_type='photron',
               rename_str=None):
    """
    Convenience function to move files/folders from a single folder into
    corresponding experiment trial folders.

    The way I store data right now, a folder is created for each ephys
    recording (axo file). I sometimes have other associated files that want
    to be in those same folders, but are captured as part of a different
    data stream, i.e. a photo of the electrode placement I take on my
    phone. This function should look at a folder of these other data files
    and deal them out into the right trial folder based on the save time

    Args:
        expr_folder: path to experiment folder containing axo (.abf) files
            should look like "XX_YYYMMDD"
        to_deal_dir: directory list containing files that need to be
            distributed to trial folders
        file_type: string ('photron', 'google_photos', 'general')
            a hacky way to get timestamps in different ways
        rename_str: string. when we copy files, rename them to whatever
            rename_str is. if None, don't rename
    """
    # get info on axo files, including date of creation
    axo_dir = sorted(glob.glob(os.path.join(expr_folder, '*', '*.abf')))
    axo_creation_times = [get_abf_tstamp(adm) for adm in axo_dir]

    # get creation times for the files to sort -- the approach will be different depending on file type
    to_deal_dir = sorted(to_deal_dir)
    if file_type == 'photron':
        to_deal_fns = [os.path.splitext(os.path.basename(file_path))[0] for file_path in to_deal_dir]
        to_deal_tstamps = [vid_filename_to_tstamp(fn) for fn in to_deal_fns]

    elif file_type == 'google_photos':
        # NB: this assumes we've saved them as jpegs. HEICs will throw us off
        to_deal_tstamps = [get_jpg_tstamp(file_path) for file_path in to_deal_dir]

    else:
        # if we're not doing anything special, just get creation time
        to_deal_tstamps = [os.path.getctime(file_path) for file_path in to_deal_dir]


    # turn the axo time stamps into bins, which we can use to sort files
    axo_bins = axo_creation_times.copy()
    final_abf = pyabf.ABF(axo_dir[-1])
    final_abf_duration = final_abf.dataLengthSec
    axo_bins.append(final_abf_duration + axo_bins[-1] + 300)  # adding a little padding to get snapshots, etc

    to_deal_ind = np.digitize(np.asarray(to_deal_tstamps), bins=axo_bins) - 1

    # loop over the axo files and see if we can find a matching file. if so, move it
    for ith, path in enumerate(axo_dir):
        # folder to current axo file
        axo_folder, _ = os.path.split(path)

        # if there are photron snapshots associated with this axo file, copy those as well
        curr_ind = np.where(to_deal_ind == ith)[0]
        for ind in curr_ind:
            src_path = to_deal_dir[ind]
            src_basename = os.path.basename(src_path)
            # do we need to change the destination file name?
            if rename_str is None:
                dst_path = os.path.join(axo_folder, src_basename)
            else:
                _, src_ext = os.path.splitext(src_basename)
                dst_path = os.path.join(axo_folder, rename_str + src_ext)

            if not os.path.exists(dst_path):
                print(f'Copying {src_path} \n to {dst_path} ...')
                shutil.copy2(src_path, dst_path)


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

    # ----------------------------------------------
    # get video file info
    vid_dir = glob.glob(os.path.join(vid_path, '*/*.mraw'))
    vid_dir += glob.glob(os.path.join(vid_path, '*/*.avi'))
    vid_fns = [os.path.splitext(os.path.basename(vd))[0] for vd in vid_dir]

    # get timestamps for videos
    vid_tstamps = [vid_filename_to_tstamp(fn) for fn in vid_fns]

    # ----------------------------------------------
    # get snapshot file info
    snapshot_dir = glob.glob(os.path.join(snapshot_path, '*.bmp'))
    snapshot_fns = [os.path.splitext(os.path.basename(ss))[0] for ss in snapshot_dir]

    # get timestamps for snapshots
    snapshot_tstamps = [vid_filename_to_tstamp(fn) for fn in snapshot_fns]

    # ----------------------------------------------
    # get indices for matching vids to axo
    axo_bins = axo_creation_times.copy()
    final_abf = pyabf.ABF(axo_dir_match[-1])
    final_abf_duration = final_abf.dataLengthSec
    axo_bins.append(final_abf_duration + axo_bins[-1] + 300)  # adding a little padding to get snapshots, etc

    vid_abf_ind = np.digitize(np.asarray(vid_tstamps), bins=axo_bins) - 1
    snapshot_abf_ind = np.digitize(np.asarray(snapshot_tstamps), bins=axo_bins) - 1

    # ----------------------------------------------
    # copy things over
    # make directory for current experiment
    expr_folder = os.path.join(out_path, next_folder)
    if not os.path.exists(expr_folder):
        os.mkdir(expr_folder)

    # add README file to folder
    readme_path = os.path.join(expr_folder, 'README.txt')
    with open(readme_path, 'w') as f:
        for row in readme_rows:
            f.write('{}:\n\n'.format(row))

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
            vid_src_folder = os.path.normpath(os.path.join(vid_src_path, '..'))
            vid_dst_folder = os.path.join(curr_folder, vid_fns[cvi])

            if not os.path.exists(vid_dst_folder):
                print('Copying {} \n to {} ...'.format(vid_src_folder, vid_dst_folder))
                shutil.copytree(vid_src_folder, vid_dst_folder)

        # if there are photron snapshots associated with this axo file, copy those as well
        curr_snapshot_ind = np.where(snapshot_abf_ind == ith)[0]
        for csi in curr_snapshot_ind:
            snapshot_src_path = snapshot_dir[csi]
            snapshot_basename = os.path.basename(snapshot_src_path)
            snapshot_dst_path = os.path.join(curr_folder, snapshot_basename)

            if not os.path.exists(snapshot_dst_path):
                print(f'Copying {snapshot_src_path} \n to {snapshot_dst_path} ...')
                shutil.copy2(snapshot_src_path, snapshot_dst_path)

    print('=========== \n Done \n ===========')
