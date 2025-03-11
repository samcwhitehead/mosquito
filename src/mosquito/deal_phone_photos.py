"""
Super simple script to distribute photos taken on my phone of the electrode placement into folders

Can also register multiple images of a single fly

TODO:
    - sometimes I take multiple photos now, at different zoom levels. how to deal with this?
"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd

from mosquito.transfer_data import deal_files

# ---------------------------------------
# PARAMS
# ---------------------------------------
# where to look for data files
DATA_ROOT = '/media/sam/SamData/Mosquitoes/'
EXPR_NAME = '80_20250305'  # which folder to look at
EXPR_FOLDER = os.path.join(DATA_ROOT, EXPR_NAME)

# path to folder of undealt photos
TO_DEAL_DIR = glob.glob(os.path.join(EXPR_FOLDER, 'phone', '*.jpg'))  # files to deal

# paht to experiment log file
LOG_PATH = os.path.join(DATA_ROOT, 'experiment_log.xlsx')

# which type of photos to distribute
FILE_TYPE = 'google_photos'  # 'google_photos' | 'photron' | 'general'

# should we rename images?
RENAME_STR = None  # 'electrode_placement'

# register images?
REGISTER_FLAG = True


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
def align_images(image, template, maxFeatures=500, keepPercent=0.2, viz_flag=False):
    """
    Register an image to a template
    From https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
    """
    # convert both the input image and template to grayscale
    if len(image.shape) > 2:
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imageGray = image.copy()

    if len(template.shape) > 2:
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        templateGray = template.copy()

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # visualize?
    if viz_flag:
        # get a grayscale version of 'aligned'
        if len(image.shape) > 2:
            alignedGray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        else:
            alignedGray = aligned

        # visualize
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 5))

        stereo_unaligned = np.zeros((template.shape[0], template.shape[1], 3), dtype=imageGray.dtype)
        stereo_unaligned[..., 0] = templateGray  # rgb2gray(template)
        stereo_unaligned[..., 1] = imageGray  # rgb2gray(image)
        ax0.imshow(stereo_unaligned)
        ax0.set_axis_off()
        ax0.set_title('pre alignment overlay')

        stereo_aligned = np.zeros((template.shape[0], template.shape[1], 3), dtype=imageGray.dtype)
        stereo_aligned[..., 0] = templateGray  # rgb2gray(template)
        stereo_aligned[..., 1] = alignedGray

        ax1.imshow(stereo_aligned)
        ax1.set_axis_off()
        ax1.set_title('post alignment overlay')

    # return the aligned image
    return aligned


# -------------------------------------------------------------------------------------------------
def run_image_registration(log_df, data_root=DATA_ROOT, data_folder=EXPR_NAME, file_ext='.jpg'):
    """
    Wrapper function for align_images that loops over all images in a given folder

    """
    # get the fly number for each data folder, so we know which to register to each other
    row_idx = (log_df['Day'] == data_folder)
    log_df_expr = log_df[row_idx]
    fly_nums = log_df_expr['Fly Num']

    fly_nums_unique = np.unique(fly_nums.values)

    # loop over files and align to a common image (if same fly)
    for fly_num in fly_nums_unique:
        # loop over trials with the same fly
        trials = log_df_expr[fly_nums == fly_num]
        axo_nums = trials['Axo Num'].astype(int)
        template_image = None

        for axo_num in axo_nums:
            # find the folder for the current axo file
            axo_paths = glob.glob(os.path.join(data_root, data_folder, '*', f'*_{int(axo_num):04d}.abf'))
            if not len(axo_paths) == 1:
                continue
            else:
                axo_path = axo_paths[0]
            axo_folder, _ = os.path.split(axo_path)
            image_path = glob.glob(os.path.join(axo_folder, f'*{file_ext}'))

            if len(image_path) < 1:
                print(f'Count not find electrode image for {data_folder}, {axo_num}')
                continue

            image_path = image_path[0]

            # read image
            image = iio.imread(image_path)

            # if this is the first image for the fly, store it as a template image. otherwise, align to template
            if template_image is None:
                # assign to template
                template_image = image.copy()
                aligned = image.copy()
            else:
                # otherwise, actually align
                aligned = align_images(image, template_image, viz_flag=True)

            # save new version of image
            save_path = image_path.replace(file_ext, f'_aligned{file_ext}')
            iio.imwrite(save_path, aligned)


# ---------------------------------------
# RUN
# ---------------------------------------
if __name__ == "__main__":
    # run code to deal photos
    deal_files(EXPR_FOLDER, TO_DEAL_DIR, file_type=FILE_TYPE, rename_str=RENAME_STR)

    # register photos?
    if REGISTER_FLAG:
        # load experiment log
        log_df = pd.read_excel(LOG_PATH)

        # run function
        run_image_registration(log_df)
