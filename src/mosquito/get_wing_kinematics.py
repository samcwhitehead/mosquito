"""
Code to analyze high-speed and regular video of mosquitoes.

 TODO:
   - put angles in body coordinates (separate function?)
   - assign leading/trailing edge. This can be done with minima of wing angle diff for high speed
        but we also want it for low speed since wings don't go all the way back
   - check if any other high speed functions needed
   - I should be able to write this so it processes images serially. The only issue now is the way we're calculating
     bg, so just need to write a rolling median estimator (should be easy)

"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from scipy.interpolate import interp1d

from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import gaussian, threshold_multiotsu
from skimage.feature import canny
from skimage.measure import label, regionprops
from skimage.morphology import (isotropic_erosion, isotropic_dilation, isotropic_opening,
                                isotropic_closing, remove_small_holes, remove_small_objects)
from skimage.segmentation import clear_border
from skimage.util import invert, compare_images

try:
    from fly_tracking_frame import FlyFrame, get_angle_from_points, get_body_angle
    from read_photron import my_read_mraw
    from util import idx_by_thresh
    from process_abf import load_processed_data, save_processed_data
except ModuleNotFoundError:
    from .fly_tracking_frame import FlyFrame, get_angle_from_points, get_body_angle
    from .read_photron import my_read_mraw
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
    'left_amp': 'angles_max'
}


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
def load_video_data(folder_id, axo_num, root_path='/media/sam/SamData/Mosquitoes',
                    subfolder_str='*_{:04d}', frame_range=None, exts=['.mraw', '.avi'],
                    return_cap_flag=False):
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
        return_cap_flag: bool, hacky way to have the function return an OpenCV
            VideoCapture output when possible

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
        search_results += glob.glob(os.path.join(search_path, f'*/*{ext}'))

    # check that we can find a unique matching file
    if len(search_results) != 1:
        raise ValueError('Could not locate file in {}'.format(search_path))

    data_path_full = search_results[0]

    # load video frames
    _, final_ext = os.path.splitext(data_path_full)

    if final_ext == '.mraw':
        # if in photron raw format, read using pySciCam
        images_gray, metadata = my_read_mraw(data_path_full, frames=frame_range)

    elif final_ext == '.avi':
        # if in avi format, read using OpenCV
        metadata = None

        # create videocapture object and pull out info
        cap = cv2.VideoCapture(data_path_full)

        # make a simple metadata dict here (should update to match Photron one...)
        metadata = {'fps': cap.get(cv2.CAP_PROP_FPS),
                    'n_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

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
            scale = 1/256
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
                images[ith] = scale*frame

        # release video capture
        cap.release()

    return images, metadata


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
        idx2 = min([imgs.shape[0] - 1, ith + window_size])
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
def get_wing_edges_roi(im_clip, mask_clip, hinge_pt, roi, canny_sigma=3.0, min_area=10,
                       extra_process_flag=False):
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

    return max_angle, min_angle


# ------------------------------------------------------------------------------------------
def track_low_speed_video(imgs, fly_frame=None, wing_sides=['right', 'left'],
                          body_angle=None, bg_window=10, canny_sigma=3.0,
                          min_area=10, extra_process_flag=True, viz_flag=False):
    """
    Function to run wing tracking on a "low-speed" video (~250 fps)

    Args:
        imgs: array of video images, where dimension 0 is the frame number and the
            final dimension is color, if using color images.
        fly_frame: instance of FlyFrame object giving fly reference frame, ROIs,
            masks, etc. See fly_tracking_frame.py. If None, manually do it here
        wing_sides: names for the wings to track. Adding this as an input to allow
            single wing tracking
        body_angle: angle of the longitudinal body axis as measured in the image frame,
            clockwise from the positive x axis
        bg_window: size of one side of the rolling window used to estimate the background
        canny_sigma: size of gaussian filter to use prior to Canny edge detection
        min_area: smallest allowable area for putative edges detected via Canny method
        extra_process_flag: bool, do some extra contrast enhancement on image?
        viz_flag: bool, visualize tracking output?

    Returns:
        wing_data: dictionary containing output of wing tracking

    """
    # convert images to grayscale if need b
    if (len(imgs.shape) > 2) and (imgs.shape[-1] == 3):
        imgs_gray = rgb2gray(imgs)
    else:
        imgs_gray = imgs

    # if we don't have a manually drawn reference frame already, get one
    if fly_frame is None:
        # initialize fly frame
        fly_frame = FlyFrame()

        # get reference frame
        cap = ArrayVideoCapture(imgs.copy())
        fly_frame.run_video(cap)
        cap.release()

    # initializing storage for wing data
    wing_data = dict()
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

    # set up view window for loop over images
    if viz_flag:
        window_name = 'wing_tracking'
        cv2.namedWindow(window_name)
        colors = [(255.0, 255.0, 255.0, 255.0), (0.0, 0.0, 0.0, 255.0)]

    # loop over images
    for ith in range(imgs_gray.shape[0]):
        # read out current image
        im_gray = imgs_gray[ith].copy()

        # loop over wing sides
        for wing_side in wing_sides:
            # crop image to current roi
            roi = wing_data[wing_side]['roi']
            im_gray_crop = im_gray[roi[1]:roi[3], roi[0]:roi[2]]

            # get background for current roi
            bg_idx = slice(max([ith - bg_window, 0]), min([ith + bg_window, imgs_gray.shape[0]]))
            bg = np.median(imgs_gray[bg_idx, roi[1]:roi[3], roi[0]:roi[2]], axis=0)

            # subtract off background from current image
            im_bg_sub = compare_images(im_gray_crop, bg)

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

            # draw circles on image if visualizing
            if viz_flag:
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

    # convert angles to body coordinates
    if body_angle is None:
        body_angle = fly_frame.get_body_angle()

    wing_data['body_angle'] = body_angle
    for wing_side in wing_sides:
        if wing_side == 'right':
            wing_sign = -1
        else:
            wing_sign = 1

        for ang in ['angles_max', 'angles_min']:
            # convert from list to array
            wing_arr = np.asarray(wing_data[wing_side][ang])

            # unwrap to [0, 2*pi] range
            wing_arr[wing_arr < 0] += 2*np.pi

            # convert to angle relative to body axis (measuring from body axis normal)
            # NB: there HAS to be a smarter way to do this
            wing_arr = np.pi/2 + wing_sign*(wing_arr - body_angle)
            wing_data[wing_side][ang] = wing_arr

    # return wing data dictionary
    return wing_data


# --------------------------------------------------------------------------------------
def run_track_low_speed_video(data_folder, axo_num, chunk_size=1000, fly_frame=None,
                              wing_sides=['right', 'left'], bg_window=10, canny_sigma=3.0,
                              min_area=10, extra_process_flag=True, viz_flag=False,
                              verbose_flag=False):
    """
    Wrapper function to run track_low_speed_video on an input video

    Args:
        data_folder: which folder (indexed by experiment number) to get data from
        axo_num: trial number to get data from
        chunk_size: number of video frames to process at a time
        fly_frame: instance of FlyFrame object giving fly reference frame, ROIs,
            masks, etc. See fly_tracking_frame.py. If None, manually do it here
        wing_sides: names for the wings to track. Adding this as an input to allow
            single wing tracking
        bg_window: size of one side of the rolling window used to estimate the background
        canny_sigma: size of gaussian filter to use prior to Canny edge detection
        min_area: smallest allowable area for putative edges detected via Canny method
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
        imgs, metadata = load_video_data(data_folder, axo_num, frame_range=(0, 2*chunk_size))
        cap = ArrayVideoCapture(imgs.copy())
        fly_frame.run_video(cap)
        cap.release()
    else:
        # otherwise we still need metadata to get total frame count
        _, metadata = load_video_data(data_folder, axo_num, frame_range=(0, 1))

    # get body angle
    head_pt = fly_frame.params['body_axis']['end_pt']
    thorax_pt = fly_frame.params['body_axis']['start_pt']
    body_angle = get_body_angle(thorax_pt, head_pt)  # + np.pi

    # run video processing in chunks (too large to do the whole thing at once)
    if 'n_frames' in metadata.keys():
        n_imgs = metadata['n_frames']
    else:
        print('Complete this code! Need to put in the method for getting frame count from .mraw metadata')

    chunks = [(x, x + chunk_size if x + chunk_size < n_imgs else n_imgs)
              for x in range(0, n_imgs, chunk_size)]

    for ith, chunk in enumerate(chunks):
        # load images in current chunk
        imgs, _ = load_video_data(data_folder, axo_num, frame_range=chunk)

        # process images
        wing_data_curr = track_low_speed_video(imgs,
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
                            int(len(cam_idx)/sync_output_rate)).astype('int')

    # restrict attention to last N points, where N is the number of wing_amp points
    # (because 'cam' is high during Record AND Ready state, we have extra signal)
    align_idx = align_idx[(-1*wing_amp.size):]

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
                          sync_output_rate=0.5, wing_sides=['right', 'left'],
                          wing_vars=['amp'], wing_var_dict=WING_VAR_DICT,
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
        sync_output_rate: rate at which camera sends sync signals relative to frame
            rate. see align_kinematics_to_cam(...) above
        wing_sides: list giving the wing sides we want to include
        wing_vars: list giving variables we want to analyze. these should correspond
            to entries in wing_var_dict
        wing_var_dict: dictionary with entries that relate tracking measurements to
            variables we know about. so, for instance, it has
               'right_amp': 'angles_min'
            signifying that the measurement in tracking, angles_min, corresponds to
            the amplitude of the right wing. for the left, it's angles_max
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

    # loop over wing side, align, and convert
    for wing_side in wing_sides:
        for var in wing_vars:
            # get current side/variable pair
            wing_amp = wing_data[wing_side][wing_var_dict[f'{wing_side}_{var}']]

            # align
            wing_amp_aligned, align_idx = align_kinematics_to_cam(wing_amp, cam,
                                                                  sync_output_rate=sync_output_rate,
                                                                  viz_flag=viz_flag)

            # interpolate
            f_amp = interp1d(t[align_idx], wing_amp_aligned[align_idx], kind='nearest',
                             bounds_error=False)
            wing_amp_interp = f_amp(t)

            # add this interpolated signal to the data dict
            data[f'{wing_side}_{var}'] = wing_amp_interp

    # also add fly reference frame to data dict
    data['fly_frame'] = wing_data['fly_frame']

    # save output?
    if save_flag:
        # for now, just saving over, since we're not changing extant fields
        data_path = data['filepath_load']
        save_processed_data(data_path, data)

    # return resulting data dict
    return data


# ---------------------------------------
# CLASSES
# ---------------------------------------
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

    def release(self):
        pass


# implement a moving median filter for images
class MovingMedian:
    def __init__(self, buffer_size, img_size):
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, img_size[0], img_size[1]), dtype=np.uint8)
        self.median_image = self.buffer[0]

    # update upon the addition of a new image
    def update(self, image):
        self.buffer = np.roll(self.buffer, 1, axis=0)
        self.buffer[0] = image
        self.median_image = self.get_median_image()

    # get median image
    def get_median_image(self):
        empty_idx = (np.sum(self.buffer, axis=(1, 2)) == 0)
        median_image = np.median(self.buffer[~empty_idx], axis=0)
        return median_image


# ---------------------------------------
# MAIN
# ---------------------------------------
# Run script
if __name__ == "__main__":
    # make instance of class
    fly_frame = FlyFrame()

    # load video to test on
    return_cap_flag = False
    imgs, _ = load_video_data(50, 27, frame_range=(0, 1500),
                               return_cap_flag=return_cap_flag)
    cap = ArrayVideoCapture(imgs.copy())

    fly_frame.run_video(cap)
    cap.release()

    # run test
    wing_data = track_low_speed_video(imgs, fly_frame=fly_frame, viz_flag=True)
