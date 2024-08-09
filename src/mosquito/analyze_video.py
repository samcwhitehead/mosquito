"""
Code to analyze high-speed and regular video of mosquitoes.


"""
# ---------------------------------------
# IMPORTS
# ---------------------------------------
import os
import glob
import cv2

import numpy as np
import skimage as ski
try:
    from tracking_frame import FlyFrame
except ModuleNotFoundError:
    from tracking_frame import FlyFrame

# ---------------------------------------
# PARAMS
# ---------------------------------------


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
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
    pass
    # # filter image
    image = invert(image)
    image = gaussian(image, sigma=1)
    image = equalize_adapthist(image, kernel_size=40)