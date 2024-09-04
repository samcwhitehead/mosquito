"""
Code to analyze high-speed and regular video of mosquitoes.

 TODO:
   - filtering/interpolating nan values in kinematics
   - merge run_track_video and run_track_video_cap -- redundant!
   - better processing for low-speed tracking (currently trying
        to pre-process videos in a Jupyter notebook called
        my_downsample_video.ipynb

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
from tabnanny import verbose

import cv2

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.records import record

from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian, threshold_multiotsu
from skimage.feature import canny
from skimage.measure import label, regionprops
from skimage.morphology import (isotropic_erosion, isotropic_dilation, isotropic_opening,
                                isotropic_closing, remove_small_holes, remove_small_objects)
from skimage.segmentation import clear_border
from skimage.util import invert, compare_images, img_as_float

try:
    from kindafly import FlyFrame, get_angle_from_points, get_body_angle
    from read_photron import my_read_mraw, my_read_cih
    from util import idx_by_thresh
    from process_abf import load_processed_data, save_processed_data
except ModuleNotFoundError:
    from .kindafly import FlyFrame, get_angle_from_points, get_body_angle
    from .read_photron import my_read_mraw, my_read_cih
    from .util import idx_by_thresh
    from .process_abf import load_processed_data, save_processed_data

# ---------------------------------------
# PARAMS
# ---------------------------------------
# the basic tracking function pulls out minima and maxima of angles in fly_frame wedges.
# depending on which side wing we're viewing, each of these could correspond to the
# 'amplitude'. this dict is meant to keep that correspondence stored
WING_VAR_DICT = {
    'right_amp': 'angles_min',
    'left_amp': 'angles_max',
    'right_lead': 'angles_lead',
    'right_trail': 'angles_trail',
    'left_lead': 'angles_lead',
    'left_trail': 'angles_trail'
}

# related to above, this gives a list of variables (in the form '{wing_side}_{var}') that
# we try to align to axo data. note that they don't all have to be there, because we check
WING_VARS = ['amp', 'lead', 'trail']

# parameters for edge detection
BG_WINDOW = 20  # 10  # take this many frames on either side of an image to get background estimate
CANNY_SIGMA = 3.0  # size of gaussian filter used prior to Canny edge detection
MIN_EDGE_AREA = 15 # minimum size for edges detected in wing wedges

# parameters for assigning leading vs trailing edge in high-speed video
MIN_HEIGHT_PRCTILE = 65
MIN_PROMINENCE_FACTOR = 0.1

# assumed multiple of fps at which camera sends sync signal to axo
SYNC_OUTPUT_RATE_DEFAULT = 0.5

# parameters for interpolating aligned wing kinematics
SMOOTH_FACTOR = 1


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
# --------------------------------------------------------------------------------------
def load_video_data(folder_id, axo_num, root_path='/media/sam/SamData/Mosquitoes',
                    subfolder_str='*_{:04d}', frame_range=None, exts=['.mraw', '.avi'],
                    data_suffix='', return_cap_flag=False, just_return_fps_flag=False):
    """
    Convenience function for loading video data

     Args:
        folder_id: folder containing processed data (in form XX_YYYYMMDD).
            If just a number is given, search for the matching folder index
        axo_num: per-day index of data file
        root_path: parent folder containing set of experiment folders
        subfolder_str: format of folder name inside experiment_folder
        frame_range: tuple giving the number of frames to read. If None,
            take all
        exts: list of extensions for video files
        data_suffix: filename suffix to look for when finding video
        return_cap_flag: bool, hacky way to have the function return an OpenCV
            VideoCapture output when possible
        just_return_fps_flag: bool, hacky way of getting this function to only
            return frame rate (in fps) sometimes so we can determine how to
            proceed with analysis

    Returns:
        images: an array of video frames
        metadata: if provided, reads the metadata info from a cih file
    """

    # check input type -- if it's a two-digit number, search for folder
    if str(folder_id).isnumeric():
        expr_folders = [f for f in os.listdir(root_path)
                        if os.path.isdir(os.path.join(root_path, f))
                        and f[:2].isdigit()]
        expr_folder_inds = [int(f.split('_')[0]) for f in expr_folders]
        expr_folder = expr_folders[expr_folder_inds.index(int(folder_id))]
    else:
        expr_folder = folder_id

    # find path to data file, given info
    search_path = os.path.join(root_path, expr_folder, subfolder_str.format(axo_num))
    search_results = []
    for ext in exts:
        search_results += glob.glob(os.path.join(search_path, f'*/*{data_suffix}{ext}'))

    # check that we can find a unique matching file
    if (len(search_results) > 1) and (data_suffix == ''):
        # this is hacky, but because the original video files have no suffix, we can
        # return multiple files if we're not careful. here, account for that using the
        # fact that anything with a suffix (i.e. '_{word}' at end of filename) will
        # come after the original file alphabetically
        search_results = sorted(search_results)
    elif len(search_results) == 1:
        pass
    else:
        raise ValueError('Could not locate file in {}'.format(search_path))

    data_path_full = search_results[0]

    # get extension of search hit
    _, final_ext = os.path.splitext(data_path_full)

    # before loading frames, see if we can get recording frame rate from .cih file
    data_path_folder, _ = os.path.split(data_path_full)
    cih_results = glob.glob(os.path.join(data_path_folder, '*.cih*'))
    if len(cih_results) == 1:
        cih = my_read_cih(cih_results[0])
        record_fps = cih['Record Rate(fps)']
    else:
        print('WARNING: could not determine video frame rate')
        record_fps = None

    # return only the video frame rate?
    if just_return_fps_flag:
        return None, {'record_fps': record_fps, 'filepath': data_path_full}

    # load video frames
    if final_ext == '.mraw':
        # if in photron raw format, read using pySciCam
        images_gray, metadata = my_read_mraw(data_path_full, frames=frame_range)

        # add filepath and record fps to metadata
        metadata['filepath'] = data_path_full
        metadata['record_fps'] = record_fps

        # should we return a video capture object?
        if return_cap_flag:
            cap = ArrayVideoCapture(images_gray)
            return cap, metadata

    elif final_ext == '.avi':
        # if in avi format, read using OpenCV
        # create videocapture object and pull out info
        cap = cv2.VideoCapture(data_path_full)

        # make a simple metadata dict here (should update to match Photron one...)
        metadata = {'playback_fps': cap.get(cv2.CAP_PROP_FPS),
                    'record_fps': record_fps,
                    'n_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'filepath': data_path_full}

        # return if we just want video capture (probably, because getting images is
        # often inefficient)
        if return_cap_flag:
            return cap, metadata

        # get image dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

        # get data type for frame
        ret, frame_test = cap.read()
        if frame_test.dtype == 'uint16':
            scale = 1 / 256
        else:
            scale = 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset frame counter

        if frame_range is not None:
            # if we specified a frame range, stick to that
            n_frames = frame_range[1] - frame_range[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0] - 1)

        else:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # initialize array for images
        images = np.zeros((n_frames, height, width, 3), dtype=np.uint8)

        # get frames
        for ith in range(n_frames):
            ret, frame = cap.read()
            if ret:
                images[ith] = scale * frame

        # release video capture
        cap.release()

    return images, metadata


# --------------------------------------------------------------------------------------
def image_to_stroke_angle(wing_angle, wing_side, body_angle):
    """
    Convenience function for converting wing angle from image coordinates--measured
    clockwise relative to positive x-axis and in the (-pi, pi) range--to "stroke angle"

    Here, stroke angle is measured relative to the line perpendicular to the body axis,
    with positive numbers going towards the head and negative going towards tail

    Args:
        wing_angle: array of wing angles measured in image frame
        wing_side: string giving which wing side we're on (e.g. 'right')
        body_angle: angle of fly's long body axis, measured in image frame

    Returns:
        wing_stroke: array of wing "stroke angles"

    """
    # have to include a sign swap for right side to keep positive going to head
    if wing_side == 'right':
        wing_sign = -1
    else:
        wing_sign = 1

    # unwrap to [0, 2*pi] range
    wing_stroke = np.asarray(wing_angle.copy())
    wing_stroke[wing_stroke < 0] += 2 * np.pi

    # convert to angle relative to body axis (measuring from body axis normal)
    # NB: there HAS to be a smarter way to do this
    wing_stroke = np.pi / 2 + wing_sign * (wing_stroke - body_angle)

    return wing_stroke


# --------------------------------------------------------------------------------------
def stroke_to_image_angle(wing_stroke, wing_side, body_angle):
    """
    Convenience function for converting wing "stroke angle" to image coordinates
    (measured clockwise relative to positive x-axis and in the (-pi, pi) range)

    Here, stroke angle is measured relative to the line perpendicular to the body axis,
    with positive numbers going towards the head and negative going towards tail

    Args:
        wing_stroke: array of wing "stroke angles"
        wing_side: string giving which wing side we're on (e.g. 'right')
        body_angle: angle of fly's long body axis, measured in image frame

    Returns:
        wing_angle: array of wing angles measured in image frame

    """
    # have to include a sign swap for right side to keep positive going to head
    if wing_side == 'right':
        wing_sign = -1
    else:
        wing_sign = 1

    # unwrap to [0, 2*pi] range
    wing_angle = np.asarray(wing_stroke.copy())
    wing_angle[wing_angle > np.pi] -= 2 * np.pi

    # convert to image frame angle
    # NB: there HAS to be a smarter way to do this
    wing_angle = wing_sign*(wing_angle - np.pi/2) + body_angle

    return wing_angle


# --------------------------------------------------------------------------------------
def get_highspeed_frame_fg(image, sigma=1, kernel=40, erode_rad=20, invert_flag=True):
    """
    Quick and dirty function to pull binary mosquito mask out from image

    Basically we filter image, binarize it, remove junk, and then take
    the largest connected component

    Args:
        image: input image, should be 8 or 16bit
        sigma: size for gaussian filter
        kernel: kernel size for adaptive histogram equalization
        erode_rad: radius for morphological erosion
        invert_flag: bool, invert image?

    Returns:
        bw_fg: binary image showing foreground object
    """
    # filter image
    if invert_flag:
        image = invert(image)
    image = gaussian(image, sigma=sigma)
    image = equalize_adapthist(image, kernel_size=kernel)

    # threshold
    thresholds = threshold_multiotsu(image, 2)
    threshold = thresholds[-1]
    bw = image > threshold

    # morphology
    dilate_rad = erode_rad + 5

    bw_mask = isotropic_erosion(bw, erode_rad)
    bw_no_border = clear_border(bw_mask)
    bw_mask = bw_mask & ~bw_no_border
    bw_mask = isotropic_dilation(bw_mask, dilate_rad)
    bw = np.asarray(~bw_mask) * bw

    # take the largest CC
    bw_label = label(bw.astype(int), connectivity=2)
    props = regionprops(bw_label)

    areas = [p['area'] for p in props]
    max_ind = np.argmax(np.asarray(areas))

    bw_out = np.zeros_like(bw)
    coords = props[max_ind]['coords']
    bw_out[coords[:, 0], coords[:, 1]] = True

    return bw_out


# --------------------------------------------------------------------------------------
def get_wing_imgs(imgs_fg, window_size=20, open_rad=1, close_rad=10,
                  min_hole_size=2000, min_obj_size=300, viz_flag=False):
    """
    Function to take a set of foreground images and extract the wings (moving) vs
    fixed (body, tether, electrodes) regions

    """
    # initialize storage for images
    imgs_fixed = np.zeros_like(imgs_fg)  # body, tether, electrodes
    imgs_moving = np.zeros_like(imgs_fg)  # wings

    # open windows if visualizing
    if viz_flag:
        wings_name = 'moving'
        fixed_name = 'fixed'
        cv2.namedWindow(wings_name)
        cv2.namedWindow(fixed_name)

    for ith in range(imgs_fg.shape[0]):
        # fixed pixels are just present in all images
        idx1 = max([0, ith - window_size])
        idx2 = min([imgs_fg.shape[0] - 1, ith + window_size])
        fixed = np.all(imgs_fg[idx1:idx2], axis=0)

        # to get moving pixels, do some extra processing
        moving = np.logical_xor(imgs_fg[ith], fixed)
        moving = isotropic_opening(moving, open_rad)
        moving = isotropic_closing(moving, close_rad)
        moving = remove_small_holes(moving, area_threshold=min_hole_size)
        moving = remove_small_objects(moving, min_size=min_obj_size)

        # store
        imgs_fixed[ith] = fixed
        imgs_moving[ith] = moving

        # visualize?
        if viz_flag:
            cv2.imshow(wings_name, 255 * moving.astype('uint8'))
            cv2.imshow(fixed_name, 255 * fixed.astype('uint8'))

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    if viz_flag:
        cv2.destroyAllWindows()

    return imgs_moving, imgs_fixed


# --------------------------------------------------------------------------------------
def initialize_wing_data(fly_frame, wing_sides=['right', 'left']):
    """
    Convenience function to begin generating a wing_data dictionary, the output of
    wing tracking

    Writing this because it looks like I may need different versions for the tracking
    code, and so having function calls will help keep versions consistent

    Args:
        fly_frame: FlyFrame object from kindafly.py
        wing_sides: list of sides of wings to analyze

    Returns:
        wing_data: dict containing fly frame info and (eventually) tracked kinematics
    """
    # create dictionary
    wing_data = dict()

    # make sure that fly_frame has updated masks and rois
    fly_frame.update_masks()

    # loop over wing sides (right, left) and fill
    for wing_side in wing_sides:
        # dictionary per side
        wing_data[wing_side] = dict()

        # empty lists for storage
        wing_data[wing_side]['angles_max'] = list()
        wing_data[wing_side]['angles_min'] = list()

        # hinge
        wing_data[wing_side]['hinge_pt'] = fly_frame.params[f'{wing_side}_wing']['hinge_pt']  # hinge point

        # roi
        roi = getattr(fly_frame, f'{wing_side}_wing').roi  # roi
        wing_data[wing_side]['roi'] = roi

        # mask
        mask = getattr(fly_frame, f'{wing_side}_wing').mask  # mask
        mask = (mask > 0)
        wing_data[wing_side]['mask'] = mask
        wing_data[wing_side]['mask_crop'] = mask[roi[1]:roi[3], roi[0]:roi[2]]

        # radii
        radius_inner = fly_frame.params[f'{wing_side}_wing']['radius_inner']  # radii of ROI
        radius_outer = fly_frame.params[f'{wing_side}_wing']['radius_outer']
        wing_data[wing_side]['radius_inner'] = radius_inner
        wing_data[wing_side]['radius_outer'] = radius_outer
        wing_data[wing_side]['radius_avg'] = (radius_inner + radius_outer) / 2.0

    return wing_data


# --------------------------------------------------------------------------------------
def get_wing_edges_roi(im_clip, mask_clip, hinge_pt, roi, canny_sigma=CANNY_SIGMA,
                       min_area=MIN_EDGE_AREA, extra_process_flag=False, debug_flag=False):
    """
    Function to extract edges from an image of the wings *IN ROI REGION*

    Args:
        im_clip: grayscale image cropped to ROI that we want to process
        mask_clip: binary mask cropped to ROI
        hinge_pt: origin of coordinate system
        roi: 4 element roi vector
        canny_sigma: gaussian filter width prior to canny edge detection
        min_area: smallest possible blob to consider in edge image
        extra_process_flag: boolean, do some additional intensity rescaling to image?
        debug_flag: boolean, visualize edge detection?

    Returns:
        angle_max, angle_min: max and min angles of detected edges in image coordinates

    """
    # scale image intensity? particularly helpful with lowspeed video
    if extra_process_flag:
        im_clip = rescale_intensity(im_clip * isotropic_dilation(mask_clip, 2))

    # get edge image (masked)
    edges = canny(im_clip, sigma=canny_sigma)
    edges *= mask_clip
    edges = isotropic_closing(edges, 3)

    # find edges as connected components
    props = regionprops(label(edges))
    props = [p for p in props if p.area > min_area]

    # get angles of each connected component relative to hinge point
    edge_angles = list()
    for p in props:
        # centroid of current component
        cm = p.centroid

        # note that we need to switch order of centroid coordinates
        edge_angles.append(get_angle_from_points(hinge_pt, np.array([cm[1] + roi[0], cm[0] + roi[1]])))

    # pull out max and min angles
    if len(edge_angles) == 0:
        max_angle = np.nan
        min_angle = np.nan

        print('No edges detected')

    elif len(edge_angles) == 1:
        max_angle = edge_angles[0]
        min_angle = np.nan

    else:
        max_angle = max(edge_angles)
        min_angle = min(edge_angles)

    # visualize?
    if debug_flag:
        # create window
        win_name = 'debug edge detection'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # create copy of input image and edge image
        im_debug = im_clip.copy()
        edge_debug = edges.copy()
        for p in props:
            cm = p.centroid
            cv2.circle(im_debug, (int(cm[1]), int(cm[0])), 4, (255, 255, 255, 255), -1)

        # show image
        while True:
            cv2.imshow(win_name, np.concatenate((im_debug, edge_debug), axis=1))

            k = cv2.waitKey() & 0xff
            if k == 27:
                break

        cv2.destroyWindow(win_name)

    return max_angle, min_angle


# ------------------------------------------------------------------------------------------
def track_video_cap(cap, fly_frame=None, wing_sides=['right', 'left'],
                    body_angle=None, bg_window=BG_WINDOW, canny_sigma=CANNY_SIGMA,
                    min_area=MIN_EDGE_AREA, extra_process_flag=True, viz_flag=False):
    """
    Function to run wing tracking on a "low-speed" video (~250 fps) using a VideoCapture

    Args:
        cap: OpenCV VideoCapture object
        fly_frame: instance of FlyFrame object giving fly reference frame, ROIs,
            masks, etc. See kindafly.py. If None, manually do it here
        wing_sides: names for the wings to track. Adding this as an input to allow
            single wing tracking
        body_angle: angle of the longitudinal body axis as measured in the image frame,
            clockwise from the positive x-axis
        bg_window: size of one side of the rolling window used to estimate the background
        canny_sigma: size of gaussian filter to use prior to Canny edge detection
        min_area: smallest allowable area for putative edges detected via Canny method
        extra_process_flag: bool, do some extra contrast enhancement on image?
        viz_flag: bool, visualize tracking output?

    Returns:
        wing_data: dictionary containing output of wing tracking

    Notes:
        Previously had bg_window=10
    """
    # if we don't have a manually drawn reference frame already, get one
    if fly_frame is None:
        # initialize fly frame
        fly_frame = FlyFrame()

        # get reference frame
        fly_frame.run_video(cap)

        # return cap to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # initializing storage for wing data
    wing_data = initialize_wing_data(fly_frame, wing_sides=wing_sides)

    # set up view window for loop over images
    if viz_flag:
        window_name = 'wing_tracking'
        cv2.namedWindow(window_name)
        colors = [(255.0, 255.0, 255.0, 255.0), (0.0, 0.0, 0.0, 255.0)]

    # also initialize background models (one for each ROI)
    bg_dict = dict()
    for wing_side in wing_sides:
        roi = wing_data[wing_side]['roi']
        roi_size = (roi[3] - roi[1], roi[2] - roi[0])
        bg_dict[wing_side] = MovingMaxImage(buffer_size=bg_window,
                                            img_size=roi_size,
                                            dtype=np.uint8)

    # loop over images
    while True:
        # read out current image
        ret, frame = cap.read()
        if not ret:
            break

        # convert to grayscale if needed
        if (len(frame.shape) > 2) and (frame.shape[-1] == 3):
            # im_gray = rgb2gray(frame)
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = frame.copy()

        # loop over wing sides
        for wing_side in wing_sides:
            # crop image to current roi
            roi = wing_data[wing_side]['roi']
            im_gray_crop = im_gray[roi[1]:roi[3], roi[0]:roi[2]]

            # subtract off background from current image
            # im_bg_sub = compare_images(im_gray_crop, bg_clip)
            # im_bg_sub = cv2.absdiff(im_gray_crop, bg_dict[wing_side].filter_image)

            im_bg_sub = cv2.subtract(bg_dict[wing_side].filter_image, im_gray_crop)

            # get edges for angles
            angle_max, angle_min = get_wing_edges_roi(im_bg_sub,
                                                      wing_data[wing_side]['mask_crop'],
                                                      wing_data[wing_side]['hinge_pt'],
                                                      roi,
                                                      canny_sigma=canny_sigma,
                                                      min_area=min_area,
                                                      extra_process_flag=extra_process_flag)

            # append these to lists
            wing_data[wing_side]['angles_max'].append(angle_max)
            wing_data[wing_side]['angles_min'].append(angle_min)

            # update background model
            bg_dict[wing_side].update(im_gray_crop)

            # draw circles on image if visualizing
            if viz_flag:
                # TEMP -- look at bg subtracted ROI
                im_gray[roi[1]:roi[3], roi[0]:roi[2]] = im_bg_sub
                for jth, angle in enumerate([angle_max, angle_min]):
                    if np.isnan(angle):
                        continue
                    pt = (wing_data[wing_side]['hinge_pt'] + (wing_data[wing_side]['radius_avg'] *
                                                              np.array([np.cos(angle), np.sin(angle)])))
                    cv2.circle(im_gray, pt.astype('int'), 4, colors[jth], -1)

        # visualize image with detected angles
        if viz_flag:
            cv2.imshow(window_name, im_gray)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    # destroy visualize windows
    if viz_flag:
        cv2.destroyAllWindows()

    # make sure we have body angle value
    if body_angle is None:
        body_angle = fly_frame.get_body_angle()

    wing_data['body_angle'] = body_angle

    # convert angles to body coordinates
    for wing_side in wing_sides:
        for ang in ['angles_max', 'angles_min']:
            wing_data[wing_side][ang] = image_to_stroke_angle(wing_data[wing_side][ang],
                                                              wing_side, body_angle)

    # return wing data dictionary
    return wing_data


# --------------------------------------------------------------------------------------
def run_track_video(data_folder, axo_num, chunk_size=1000, fly_frame=None,
                    wing_sides=['right', 'left'], bg_window=BG_WINDOW,
                    canny_sigma=CANNY_SIGMA, min_area=MIN_EDGE_AREA,
                    data_suffix='', extra_process_flag=True,
                    viz_flag=False, verbose_flag=False):
    """
    Wrapper function to run track_video on an input video

    Args:
        data_folder: which folder (indexed by experiment number) to get data from
        axo_num: trial number to get data from
        chunk_size: number of video frames to process at a time
        fly_frame: instance of FlyFrame object giving fly reference frame, ROIs,
            masks, etc. See kindafly.py. If None, manually do it here
        wing_sides: names for the wings to track. Adding this as an input to allow
            single wing tracking
        bg_window: size of one side of the rolling window used to estimate the background
        canny_sigma: size of gaussian filter to use prior to Canny edge detection
        min_area: smallest allowable area for putative edges detected via Canny method
        data_suffix: string, filename suffix to look for when loading video data
        extra_process_flag: bool, do some extra contrast enhancement on image?
        viz_flag: bool, visualize tracking output?
        verbose_flag: bool, print updates during processing?

    Returns:
        wing_data: a dictionary containing measured wing angles and fly frame

    """
    # name for angle variables
    angle_var_names = ['angles_max', 'angles_min']
    wing_data = dict()

    # get fly reference frame if we don't have it already
    if fly_frame is None:
        # initialize fly frame
        fly_frame = FlyFrame()

        # allow user to set frame coordinates
        cap, metadata = load_video_data(data_folder, axo_num, frame_range=(0, 2 * chunk_size),
                                         data_suffix=data_suffix, return_cap_flag=True)
        fly_frame.run_video(cap)
        cap.release()
    else:
        # otherwise we still need metadata to get total frame count
        _, metadata = load_video_data(data_folder, axo_num, frame_range=(0, 1),
                                      data_suffix=data_suffix)

    # get body angle
    head_pt = fly_frame.params['body_axis']['end_pt']
    thorax_pt = fly_frame.params['body_axis']['start_pt']
    body_angle = get_body_angle(thorax_pt, head_pt)  # + np.pi

    # run video processing in chunks (too large to do the whole thing at once)
    if 'n_frames' in metadata.keys():
        n_imgs = metadata['n_frames']
    elif 'Total Frame' in metadata.keys():
        n_imgs = metadata['Total Frame']
    else:
        raise Exception('Cannot locate total frame count!')

    chunks = [(x, x + chunk_size if x + chunk_size < n_imgs else n_imgs)
              for x in range(0, n_imgs, chunk_size)]

    for ith, chunk in enumerate(chunks):
        # load images in current chunk
        imgs, _ = load_video_data(data_folder, axo_num, frame_range=chunk, data_suffix=data_suffix)
        cap = ArrayVideoCapture(imgs.copy())

        # process images
        wing_data_curr = track_video_cap(cap,
                                         fly_frame=fly_frame,
                                         wing_sides=wing_sides,
                                         body_angle=body_angle,
                                         bg_window=bg_window,
                                         canny_sigma=canny_sigma,
                                         min_area=min_area,
                                         extra_process_flag=extra_process_flag,
                                         viz_flag=viz_flag)

        # store data
        if ith == 0:
            wing_data = wing_data_curr
        else:
            for wing_side in wing_sides:
                for var in angle_var_names:
                    wing_data[wing_side][var] = np.concatenate((wing_data[wing_side][var],
                                                                wing_data_curr[wing_side][var]))

        if verbose_flag:
            print(f'Completed processing frames {chunk[0]} through {chunk[1]}')

    # finish up
    wing_data['fly_frame'] = fly_frame.__dict__
    wing_data['record_fps'] = metadata['record_fps']

    return wing_data


# --------------------------------------------------------------------------------------
def run_track_video_cap(data_folder, axo_num, fly_frame=None,
                        wing_sides=['right', 'left'], bg_window=BG_WINDOW,
                        canny_sigma=CANNY_SIGMA, min_area=MIN_EDGE_AREA,
                        data_suffix='', extra_process_flag=True,
                        viz_flag=False, verbose_flag=False):
    """
    Wrapper function to run track_video on an input video using VideoCapture

    Args:
        data_folder: which folder (indexed by experiment number) to get data from
        axo_num: trial number to get data from
        fly_frame: instance of FlyFrame object giving fly reference frame, ROIs,
            masks, etc. See kindafly.py. If None, manually do it here
        wing_sides: names for the wings to track. Adding this as an input to allow
            single wing tracking
        bg_window: size of one side of the rolling window used to estimate the background
        canny_sigma: size of gaussian filter to use prior to Canny edge detection
        min_area: smallest allowable area for putative edges detected via Canny method
        data_suffix: string, filename suffix to look for when loading video data
        extra_process_flag: bool, do some extra contrast enhancement on image?
        viz_flag: bool, visualize tracking output?
        verbose_flag: bool, print updates during processing?

    Returns:
        wing_data: a dictionary containing measured wing angles and fly frame

    """
    # load VideoCapture for video
    cap, metadata = load_video_data(data_folder, axo_num, return_cap_flag=True,
                                    data_suffix=data_suffix)

    # get fly reference frame if we don't have it already
    if fly_frame is None:
        # initialize fly frame
        fly_frame = FlyFrame()

        # let user set coordinates
        fly_frame.run_video(cap)

        # return cap to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # get body angle
    head_pt = fly_frame.params['body_axis']['end_pt']
    thorax_pt = fly_frame.params['body_axis']['start_pt']
    body_angle = get_body_angle(thorax_pt, head_pt)  # + np.pi

    # process images
    wing_data = track_video_cap(cap,
                                fly_frame=fly_frame,
                                wing_sides=wing_sides,
                                body_angle=body_angle,
                                bg_window=bg_window,
                                canny_sigma=canny_sigma,
                                min_area=min_area,
                                extra_process_flag=extra_process_flag,
                                viz_flag=viz_flag)

    # print update?
    if verbose_flag:
        print(f'Completed processing frames')

    # finish up
    wing_data['fly_frame'] = fly_frame.__dict__
    wing_data['record_fps'] = metadata['record_fps']
    cap.release()

    return wing_data


# --------------------------------------------------------------------------------------
def assign_leading_trailing(angles_min, angles_max,
                            min_height_prctile=MIN_HEIGHT_PRCTILE,
                            min_prom_factor=MIN_PROMINENCE_FACTOR,
                            viz_flag=False):
    """
    Function to get leading and trailing edge angles from the two angles we measure
    directly from video: min and max angles (corresponding to the extremes
    of the wing wedge ROI edge angles).

    *This function assumes that angles_min and angles_max are in stroke angle coords
    (see image_to_stroke_angle function in this file)

    NB: this is really only applicable for high-speed video, since we can only get the
    front of the wing envelope in low-speed

    Args:
        angles_min: array of minimum angles measured from video IN STROKE ANGLE COORDS
        angles_max: array of maximum angles measured from video IN STROKE ANGLE COORDS
        min_height_prctile: percentile value used to calculate minimum height in peak
            detection, which is used to identify wing flip points
        min_prom_factor: value multiplied by the range (max - min) of the signal to
            determine the minimum peak prominence, used to identify wing flip points
        viz_flag: boolean, visualize leading/trailing assignment?

    Returns:
        angles_leading: array of angles of the wing LEADING edge (will be in stroke
            angle coordinates, like the input)
        angles_trailing: array of angles of the wing TRAILING edge (will be in stroke
            angle coordinates, like the input)

    """
    # get signals for the difference between the two angles at all times
    wing_angle_diff = angles_max - angles_min

    # 'max' and 'min' labels are assigned when we extract angles in image frame, so
    # left vs right side will have a sign flip. remove that for consistency
    if np.nanmean(wing_angle_diff) < 0:
        wing_angle_diff *= -1
        angles_max_copy = angles_max.copy()
        angles_max = angles_min
        angles_min = angles_max_copy

    # get signals for wing "center of mass" (in angle coordinates) and its velocity
    wing_cm_angles = (np.asarray(angles_max) + np.asarray(angles_min))/2.0
    wing_cm_vel = np.gradient(wing_cm_angles)

    # wing flipping happens at minima of wing_angle_diff signal. locate these points
    min_height = np.nanpercentile(-1*wing_angle_diff, min_height_prctile)
    min_prominence = min_prom_factor * (np.nanmax(wing_angle_diff) -
                                        np.nanmin(wing_angle_diff))

    flips, _ = find_peaks(-1*wing_angle_diff,
                          height=(min_height, None),
                          prominence=(min_prominence, None))

    # get indices for chunks between flips to find avg vel in these regions
    chunk_idx = [np.sum(t >= flips) for t in np.arange(wing_angle_diff.size)]
    chunk_idx = np.asarray(chunk_idx)
    chunk_idx_unique = np.unique(chunk_idx)

    angles_leading = []  # initialize some storage
    angles_trailing = []

    # loop over chunks, get average angular velocity, and assign leading/trailing
    for chunk in chunk_idx_unique:
        idx = (chunk_idx == chunk)
        vel_mean = np.mean(wing_cm_vel[idx])

        if vel_mean <= 0:
            leading = angles_min[idx]
            trailing = angles_max[idx]
        else:
            leading = angles_max[idx]
            trailing = angles_min[idx]

        angles_leading.extend(leading)
        angles_trailing.extend(trailing)

    # convert to arrays
    angles_leading = np.asarray(angles_leading)
    angles_trailing = np.asarray(angles_trailing)

    # now go back and examine first and final chunks. because they're probably
    # incomplete, their average velocity may have the wrong sign
    check_pair = [[0, 1], [-1, -2]]  # each entry list contains index pairs to check
    for pair in check_pair:
        # get indices for chunks to compare
        idx_check = (chunk_idx == chunk_idx_unique[pair[0]])  # chunk to check
        idx_ref = (chunk_idx == chunk_idx_unique[pair[1]])  # chunk to reference

        # compare velocities -- they should have opposite sign. if not, swap
        vel_check_sign = np.sign(np.mean(wing_cm_vel[idx_check]))
        vel_ref_sign = np.sign(np.mean(wing_cm_vel[idx_ref]))
        if vel_check_sign == vel_ref_sign:
            tmp = angles_leading[idx_check].copy()
            angles_leading[idx_check] = angles_trailing[idx_check]
            angles_trailing[idx_check] = tmp

    # visualize results?
    if viz_flag:
        # make figure
        fig, ax = plt.subplots()

        # plot angles
        ax.plot(angles_leading, 'b-', label='leading')
        ax.plot(angles_trailing, 'r-', label='trailing')

        # label axes
        ax.set_xlabel('time (index)')
        ax.set_ylabel('angle (rad)')
        ax.legend()

        # display figure
        plt.show()

    # return angles
    return angles_leading, angles_trailing, flips


# --------------------------------------------------------------------------------------
def run_assign_leading_trailing(wing_data, wing_sides=['right', 'left'],
                                min_height_prctile=MIN_HEIGHT_PRCTILE,
                                min_prom_factor=MIN_PROMINENCE_FACTOR,
                                viz_flag=False):
    """
    Wrapper function to run 'assign_leading_trailing' on a wing_data dictionary (the
    output of run_track_video)

    Args:
        wing_data: dictionary containing wing data; output of run_track_video and
            run_track_video_cap
        wing_sides: list with strings indicating wing sides
        min_height_prctile: percentile value used to calculate minimum height in peak
            detection, which is used to identify wing flip points
        min_prom_factor: value multiplied by the range (max - min) of the signal to
            determine the minimum peak prominence, used to identify wing flip points
        viz_flag: boolean, visualize leading/trailing assignment?

    Returns:
        wing_data: updated wing data dictionary

    """
    # loop over wing sides and do assignment
    for wing_side in wing_sides:
        # load angles for current wing side
        angles_min = wing_data[wing_side]['angles_min']
        angles_max = wing_data[wing_side]['angles_max']

        # do leading/trailing assignment
        angles_lead, angles_trail, flips = assign_leading_trailing(angles_min, angles_max,
                                                                   min_height_prctile=min_height_prctile,
                                                                   min_prom_factor=min_prom_factor,
                                                                   viz_flag=viz_flag)

        # add leading/trailing to dictionary; remove min/max
        wing_data[wing_side]['angles_lead'] = angles_lead
        wing_data[wing_side]['angles_trail'] = angles_trail
        wing_data[wing_side]['flip_ind'] = flips
        del wing_data[wing_side]['angles_min']
        del wing_data[wing_side]['angles_max']

    # return updated dictionary
    return wing_data


# --------------------------------------------------------------------------------------
def align_kinematics_to_cam(wing_amp, cam, sync_output_rate=0.5, viz_flag=False):
    """
    Convenience function to align measured kinematics data to 'cam' signal in abf file.

    Cam signal is tied to frame capture, but its rate is multiplied by sync_output_rate
    (e.g. if sync_output_rate is 0.5, the cam signal is only high every other frame)

    Args:
        wing_amp: measured wing signal from video data, indexed by video frame
        cam: signal from camera to DAQ, high when we're recording a frame
            (but see caveat with sync_output_rate)
        sync_output_rate: rate at which the camera sends sync signals relative
            to frame rate. E.g. if sync_output_rate=0.5, we get a high cam signal
            every other frame. If sync_output_rate=2, we get two high cam signals
            per frame
        viz_flag: boolean, do we want to visualize alignment?

    Returns:
        wing_amp_aligned: aligned wing kinematic signal (NaNs where we don't have data)
        align_idx: indices in abf (cam) signal corresponding to wing amp measurements
    """
    # find where camera signal is high
    cam_idx = idx_by_thresh(cam)

    # we sometimes get two indices, so just take the first
    cam_idx = [idx[0] for idx in cam_idx]

    # resample according to sync_output_rate
    align_idx = np.linspace(cam_idx[0], cam_idx[-1],
                            int(len(cam_idx) / sync_output_rate)).astype('int')

    # restrict attention to last N points, where N is the number of wing_amp points
    # (because 'cam' is high during Record AND Ready state, we have extra signal)
    align_idx = align_idx[(-1 * wing_amp.size):]

    # get wing kinematic variable values at sample points
    wing_amp_aligned = np.nan * np.ones(cam.shape)
    wing_amp_aligned[align_idx] = wing_amp

    # visualize?
    if viz_flag:
        fig, (ax_cam, ax_wing) = plt.subplots(2, 1, figsize=(11, 4), sharex=True)

        # plot cam signal
        ax_cam.plot(cam)
        ax_cam.set_ylabel('camera sync signal')

        # plot wing signal
        ax_wing.plot(wing_amp_aligned, '.')
        ax_wing.set_ylabel('wing data')
        ax_wing.set_xlabel('time (idx)')

        plt.show()

    return wing_amp_aligned, align_idx


# --------------------------------------------------------------------------------------
def add_kinematics_to_axo(wing_data, data_folder, axo_num, data_suffix='_processed',
                          sync_output_rate_default=SYNC_OUTPUT_RATE_DEFAULT,
                          wing_sides=['right', 'left'], wing_vars=WING_VARS,
                          wing_var_dict=WING_VAR_DICT, smooth_factor=SMOOTH_FACTOR,
                          save_flag=False, viz_flag=False):
    """
    Function to add tracked wing kinematics to processed data dictionaries containing
    ephys measurements (from mosquito/src/process_abf.py)

    To do this, need to align kinematics data (measured each video frame) with the
    'cam' signal in our abf files, which, currently, signal every other frame, and
    begin when the camera is in 'Ready' state, i.e. not just when it is recording

    Args:
        wing_data: dictionary containing tracked wing kinematics
        data_folder: expr folder corresponding to this trial (XX_YYYYMMDD form)
        axo_num: number of current trial
        data_suffix: suffix for processed data file to load
        sync_output_rate_default: assumed rate at which camera sends sync signals
            relative to its frame rate. see align_kinematics_to_cam(...) above
            NB: we'll calculate this if we have sufficient info though
        wing_sides: list giving the wing sides we want to include
        wing_vars: list giving variables we want to analyze. these should correspond
            to entries in wing_var_dict
        wing_var_dict: dictionary with entries that relate tracking measurements to
            variables we know about. so, for instance, it has
               'right_amp': 'angles_min'
            signifying that the measurement in tracking, angles_min, corresponds to
            the amplitude of the right wing. for the left, it's angles_max
        smooth_factor: scipy UnivariateSpline positive smoothing factor, used to
            choose number of spline knots for wing data interpolant
        save_flag: bool, save resultant combined data?
        viz_flag: visualize alignment? gets passed to align_kinematics_to_cam

    Returns:
        data: processed abf data dict now with kinematics

    """
    # load abf data
    data = load_processed_data(data_folder, axo_num, data_suffix=data_suffix)

    # read out signals to use for alignment
    cam = data['cam']
    t = data['time']

    # initialize a sub-dictionary in data to store wing kinematics
    if 'wing' not in data.keys():
        data['wing'] = dict()

    # try to guess sync_output_rate from 1) camera fps and 2) cam signal to axo
    if 'record_fps' in wing_data.keys():
        # get actual camera frame rate
        vid_fps = wing_data['record_fps']

        # get frequency of signal from cam to axo
        fs = data['sampling_freq']
        cam_idx = idx_by_thresh(cam)
        cam_idx = np.asarray([idx[0] for idx in cam_idx])
        cam_freq = fs/np.mean(np.diff(cam_idx))

        # use these two to get sync_output_rate
        # NB: possible values are 0.5, 1, 2, 4, so we need a round step
        sync_output_rate = round(10 * (cam_freq / vid_fps)) / 10.0

    else:
        # otherwise just use default guess
        sync_output_rate = sync_output_rate_default

    # should only need to do alignment once, since video measurements all per frame
    align_idx = None

    # loop over wing side, align, and convert
    for wing_side in wing_sides:
        for var in wing_vars:
            # get current side/variable pair
            if wing_var_dict[f'{wing_side}_{var}'] not in wing_data[wing_side].keys():
                continue
            wing_amp = wing_data[wing_side][wing_var_dict[f'{wing_side}_{var}']]

            # align if we haven't already
            if align_idx is None:
                wing_amp_aligned, align_idx = align_kinematics_to_cam(wing_amp, cam,
                                                                      sync_output_rate=sync_output_rate,
                                                                      viz_flag=viz_flag)

            else:
                # otherwise just use previously computed values
                wing_amp_aligned = np.nan * np.ones(cam.shape)
                wing_amp_aligned[align_idx] = wing_amp

            # interpolate
            wing_amp_interp = wing_amp_aligned.copy()

            align_nan_idx = np.isnan(wing_amp_aligned[align_idx])
            f_amp = UnivariateSpline(t[align_idx][~align_nan_idx],
                                     wing_amp_aligned[align_idx][~align_nan_idx],
                                     s=smooth_factor,
                                     ext=3)  # ext=3 means we extrapolate using boundary vals

            # to preserve nans outside video region, only apply interpolant to small region
            interp_region = np.arange(align_idx[~align_nan_idx][0],
                                      align_idx[~align_nan_idx][-1])
            wing_amp_interp[interp_region] = f_amp(t[interp_region])

            # add this interpolated signal to the data dict
            data['wing'][f'{wing_side}_{var}_raw'] = wing_amp_aligned
            data['wing'][f'{wing_side}_{var}'] = wing_amp_interp

    # also add fly reference frame to data dict
    data['wing']['fly_frame'] = wing_data['fly_frame']

    # save output?
    if save_flag:
        # for now, just saving over, since we're not changing extant fields
        data_path = data['filepath_load']
        save_processed_data(data_path, data)

    # return resulting data dict
    return data


# --------------------------------------------------------------------------------------
def calc_stroke_amplitude(angles_lead):
    """
    Should make this a little more general, but add a function to get stroke amplitude
    for kinematics from high-speed video, based on the angle of the leading edge

    Args:
        angles_lead: angle of leading edge of the wing, in body (aka stroke) coordinates

    Returns:
        stroke_amp: stroke amplitude, in whichever units angles_lead is in
        stroke_amp_ind: indices in ephys data array where we measure stroke amplitude
            (halfway between the extrema, so we get two measurments per wingstroke)
        stroke_max_ind: indices where stroke is at its maximum
        stroke_min_ind: indices where stroke is at its minimum

    """
    # fit spline to angle input (where it isn't nan)
    nan_idx = np.isnan(angles_lead)
    t_samp = np.arange(angles_lead.size)

    angle_spline = UnivariateSpline(t_samp[~nan_idx], angles_lead[~nan_idx],
                                    s=0, k=4, ext=3)

    # find roots of derivative of spline
    angle_spline_deriv = angle_spline.derivative()
    extrema_times = angle_spline_deriv.roots()
    extrema_ind = np.asarray([np.argmin(np.abs(t_samp[~nan_idx] - et))
                              for et in extrema_times])
    extrema_ind += np.where(~nan_idx)[0][0]

    # determine points of max/min stroke angle
    angles_lead_mean = np.nanmean(angles_lead)
    stroke_max_ind = extrema_ind[angles_lead[extrema_ind] > angles_lead_mean]
    stroke_min_ind = extrema_ind[angles_lead[extrema_ind] < angles_lead_mean]

    # calculate stroke amplitude
    stroke_amp = np.abs(np.diff(angles_lead[extrema_ind]))
    stroke_amp_ind = (extrema_ind[:-1] + extrema_ind[1:]) / 2
    stroke_amp_ind = stroke_amp_ind.astype('int')

    # return
    return stroke_amp, stroke_amp_ind, stroke_max_ind, stroke_min_ind


# --------------------------------------------------------------------------------------
def analyze_video(data_folder, axo_num, fly_frame=None, data_suffix='',
                  wing_sides=['right', 'left'], bg_window=BG_WINDOW,
                  canny_sigma=CANNY_SIGMA, min_area=MIN_EDGE_AREA,
                  extra_process_flag=True, min_height_prctile=MIN_HEIGHT_PRCTILE,
                  min_prom_factor=MIN_PROMINENCE_FACTOR, axo_data_suffix='_processed',
                  sync_output_rate_default=SYNC_OUTPUT_RATE_DEFAULT,
                  wing_vars=WING_VARS, wing_var_dict=WING_VAR_DICT,
                  smooth_factor=SMOOTH_FACTOR, save_flag=False,
                  viz_flag=False, verbose_flag=False, make_plot_flag=False):
    """
    Wrapper function to load a flight video, analyze it, and add the resulting kinematic
    measurements to the data dictionary containing ephys signals, etc.

    Args:
        data_folder: expr folder corresponding to this trial (XX_YYYYMMDD form)
            (general)

        axo_num: number of current trial (general)

        fly_frame: instance of FlyFrame object giving fly reference frame, ROIs,
            masks, etc. See kindafly.py. If None, manually do it here

        data_suffix: string, filename suffix to look for when loading video data
            (general)

        wing_sides: names for the wings to track. Adding this as an input to allow
            single wing tracking (general)

        bg_window: size of one side of the rolling window used to estimate the background
            (run_track_video_cap)

        canny_sigma: size of gaussian filter to use prior to Canny edge detection
            (run_track_video_cap)

        min_area: smallest allowable area for putative edges detected via Canny method
            (run_track_video_cap)

        extra_process_flag: bool, do some extra contrast enhancement on image?
            (run_track_video_cap)

        min_height_prctile: percentile value used to calculate minimum height in peak
            detection, which is used to identify wing flip points
            (run_assign_leading_trailing)

        min_prom_factor: value multiplied by the range (max - min) of the signal to
            determine the minimum peak prominence, used to identify wing flip points
            (run_assign_leading_trailing)

        axo_data_suffix: data suffix for axo data file to load (add_kinematics_to_axo)

        sync_output_rate_default: assumed rate at which camera sends sync signals
            relative to its frame rate. see align_kinematics_to_cam(...) above
            NB: we'll calculate this if we have sufficient info though
            (add_kinematics_to_axo)

        wing_vars: list giving variables we want to analyze. these should correspond
            to entries in wing_var_dict (add_kinematics_to_axo)

        wing_var_dict: dictionary with entries that relate tracking measurements to
            variables we know about (add_kinematics_to_axo)

        smooth_factor: scipy UnivariateSpline positive smoothing factor, used to
            choose # of spline knots for wing data interp (add_kinematics_to_axo)

        save_flag: bool,  save data dictionary with ephys + aligned kinematics?
            (add_kinematics_to_axo)

        viz_flag: bool, visualize tracking output? (general)

        verbose_flag: bool, print updates during processing? (general)

        make_plot_flag: bool, generate a plot of ~stroke angle and save, so we have
            an easy check for goodness of tracking?

    Returns:
        data: dictionary containing ephys data and aligned kinematic data

        wing_data: dictionary containing wing kinematic data

    """
    # print an update?
    if verbose_flag:
        print(f'Beginning tracking for expr {data_folder}, axo {axo_num}...')

    # run wing kinematic extraction
    wing_data = run_track_video_cap(data_folder, axo_num,
                                    fly_frame=fly_frame,
                                    wing_sides=wing_sides,
                                    bg_window=bg_window,
                                    canny_sigma=canny_sigma,
                                    min_area=min_area,
                                    data_suffix=data_suffix,
                                    extra_process_flag=extra_process_flag,
                                    viz_flag=viz_flag,
                                    verbose_flag=verbose_flag)

    # print an update?
    if verbose_flag:
        print(f'Completed tracking for expr {data_folder}, axo {axo_num}')
        print(f'Processing kinematics for expr {data_folder}, axo {axo_num}...')

    # assign leading vs trailing edge, which is different for high- vs low-speed data
    highspeed_cutoff = 2000  # cutoff for guessing whether current vid is high-speed
    record_fps = wing_data['record_fps']
    if record_fps is None:
        record_fps = 0  # just avoid annoying exceptions here
    if record_fps > highspeed_cutoff:
        wing_data = run_assign_leading_trailing(wing_data,
                                                wing_sides=wing_sides,
                                                min_height_prctile=min_height_prctile,
                                                min_prom_factor=min_prom_factor,
                                                viz_flag=viz_flag)

    else:
        # for low-speed, basically just switching around labels, but max/min map to
        # leading/trailing edges differently depending on wing side
        for wing_side in wing_sides:
            if wing_side == 'right':
                wing_data[wing_side]['angles_lead'] = wing_data[wing_side]['angles_min']
                wing_data[wing_side]['angles_trail'] = wing_data[wing_side]['angles_max']
            elif wing_side == 'left':
                wing_data[wing_side]['angles_lead'] = wing_data[wing_side]['angles_max']
                wing_data[wing_side]['angles_trail'] = wing_data[wing_side]['angles_min']

            del wing_data[wing_side]['angles_min']
            del wing_data[wing_side]['angles_max']

    # align wing kinematics to axoscope data
    data = add_kinematics_to_axo(wing_data, data_folder, axo_num,
                                 data_suffix=axo_data_suffix,
                                 sync_output_rate_default=sync_output_rate_default,
                                 wing_sides=wing_sides,
                                 wing_vars=wing_vars,
                                 wing_var_dict=wing_var_dict,
                                 smooth_factor=smooth_factor,
                                 save_flag=save_flag,
                                 viz_flag=viz_flag)

    # add video frame rate to data dictionary
    data['wing']['record_fps'] = record_fps

    # also get stroke amplitude measurements, if video is high-speed
    if record_fps > highspeed_cutoff:
        # loop over wing sides to get values
        for wing_side in wing_sides:
            # read out current leading edge angle
            angles_lead = data['wing'][f'{wing_side}_lead']

            # calculate stroke amplitudes
            (stroke_amp, stroke_amp_ind, stroke_max_ind, stroke_min_ind) = (
                calc_stroke_amplitude(angles_lead))

            # store
            data['wing'][f'{wing_side}_stroke_amp'] = stroke_amp
            data['wing'][f'{wing_side}_stroke_amp_ind'] = stroke_amp_ind
            data['wing'][f'{wing_side}_stroke_max_ind'] = stroke_max_ind
            data['wing'][f'{wing_side}_stroke_min_ind'] = stroke_min_ind

    # make plot to show tracking output?
    if make_plot_flag:
        # initialize figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # read out some variables
        t = data['time']
        right_angle = data['wing']['right_lead']
        left_angle = data['wing']['left_lead']
        nan_ind = np.where(np.isnan(right_angle))[0]

        # plot different range depending on whether video is high-speed or no
        if record_fps > highspeed_cutoff:
            # for high-speed video, plot small range of left and right stroke
            tmin = t[nan_ind[0]]
            tmax = t[nan_ind[0]+200]

        else:
            # for low-speed, plot stroke amp for full range
            tmin = t[nan_ind[0]]
            tmax = t[nan_ind[-1]]

        # restrict plot to specified time range
        plot_mask = (t >= tmin) & (t <= tmax)

        # plot
        ax.plot(t[plot_mask], right_angle[plot_mask], 'r-', label='right')
        ax.plot(t[plot_mask], left_angle[plot_mask], 'b-', label='left')

        # label axes
        ax.legend()
        ax.set_ylabel('angle (rad)')
        ax.set_xlabel('time (s)')

        # save figure
        filepath_load = data['filepath_load']
        save_path, _ = os.path.split(filepath_load)
        fig.savefig(os.path.join(save_path, 'wing_tracking.png'))

    # print an update?
    if verbose_flag:
        print(f'Completed processing kinematics for expr {data_folder}, axo {axo_num}')
        print('Done!')

    # return axo data (with aligned kinematics) and wing_data
    return data, wing_data


# ---------------------------------------
# CLASSES
# ---------------------------------------
# ---------------------------------------------------------------------------------------
# create a VideoCapture type object from an array of images in memory
class ArrayVideoCapture:
    def __init__(self, array):
        self.array = array
        self.index = 0
        self.total_frames = len(array)

    def read(self):
        if self.index < self.total_frames:
            frame = self.array[self.index]
            self.index += 1
            return True, frame
        else:
            return False, None

    # mimic OpenCV cap.set
    def set(self, command, val):
        if command == cv2.CAP_PROP_POS_FRAMES:
            # set frame index
            self.index = val

        else:
            print('fill this out!')

    def release(self):
        pass


# ---------------------------------------------------------------------------------------
# implement a moving MEDIAN filter for images
class MovingMedianImage:
    def __init__(self, buffer_size, img_size, dtype='float64'):
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, img_size[0], img_size[1]), dtype=dtype)
        self.filter_image = self.buffer[0]

    # update upon the addition of a new image
    def update(self, image):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = image
        self.get_filter_image()

    # get median image
    def get_filter_image(self):
        empty_idx = (np.sum(self.buffer, axis=(1, 2)) == 0)
        np.median(self.buffer[~empty_idx], axis=0, out=self.filter_image)


# ---------------------------------------------------------------------------------------
# implement a moving MAXIMUM filter for images
class MovingMaxImage:
    def __init__(self, buffer_size, img_size, dtype='float64'):
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, img_size[0], img_size[1]), dtype=dtype)
        self.filter_image = self.buffer[0]

    # update upon the addition of a new image
    def update(self, image):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = image
        self.get_filter_image()

    # get median image
    def get_filter_image(self):
        np.max(self.buffer, axis=0, out=self.filter_image)


# ---------------------------------------
# MAIN
# ---------------------------------------
# Run script
if __name__ == "__main__":
    # make instance of class
    fly_frame = FlyFrame()

    # load video to test on
    return_cap_flag = True
    cap, _ = load_video_data(50, 27, frame_range=(0, 1500),
                              return_cap_flag=return_cap_flag)
    fly_frame.run_video(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # run test
    wing_data = track_video_cap(cap, fly_frame=fly_frame, viz_flag=True)
