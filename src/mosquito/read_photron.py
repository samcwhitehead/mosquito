"""
Code for handling videos output by Photron software, which are by default saved as 12-bit
images in the mraw format (this is the camera sensor depth), a difficult thing for opencv,
etc. to handle

This will mostly be a wrapper for the very useful pySciCam package, but also going to do
an adaptation of a cih file reader from pyMRAW, which seems good, but doesn't let me work
with 12-bit images
"""
import os
import warnings
import xmltodict
import glob
import cv2

import numpy as np

from pySciCam.pySciCam import ImageSequence


#########################################################################################
######################## PARAMS/CONSTANTS ###############################################
#########################################################################################
SUPPORTED_FILE_FORMATS = ['mraw', 'tiff', 'avi']
SUPPORTED_EFFECTIVE_BIT_SIDE = ['lower', 'higher']

MERGED_FPS = 50
MERGED_DISPLAY_FLAG = False

ROOT_PATH = '/media/sam/SamData/HighSpeedVideo'
FILE_EXT = '.mraw'  # file type we expect to find
FILE_ID_FORMAT = 'C%03dH%03dS%04d'
# DATE_STR_FORMAT = ''  # it looks like YYYYMMDD_HHMMSS but not sure how best to code


#########################################################################################
########################### FUNCTIONS ###################################################
#########################################################################################

def get_vid_filename(root_path=ROOT_PATH, expt_folder=None, sub_folder=None,
                     cam_num=1, header_num=1, partition_num=1, file_ext=FILE_EXT,
                     file_id_format=FILE_ID_FORMAT, date_str=None):
    """
    Using some common file identifiers, pull out the full path to a video file

    :param root_path: parent directory in which high speed data is stored
    :param expt_folder: name of folder within root_path where data is stored
    :param sub_folder: name of folder within root_path/expt_folder
    :param cam_num: camera number
    :param header_num: header number
    :param partition_num: partition number
    :param file_ext: extension for file we're looking for
    :param file_id_format: format of identifier appended to save fns
    :param date_str: string in the format YYYYMMDD_hhmmss

    :return: filename for video file
    """
    # combine file identifiers to get likely filename/folder elements
    file_id_str = file_id_format % (cam_num, header_num, partition_num)
    if not date_str is None:
        file_search_str = '*%s*/*_%s*_%s%s' % (file_id_str, file_id_str, date_str, file_ext)
    else:
        file_search_str = '*%s*/*_%s*%s' % (file_id_str, file_id_str, file_ext)

    # generate path that we'll search for data
    search_path = root_path
    if expt_folder is not None:
        search_path = os.path.join(search_path, expt_folder)
    if sub_folder is not None:
        search_path = os.path.join(search_path, sub_folder)
    search_path = os.path.join(search_path, file_search_str)

    # perform search
    possible_hits = glob.glob(search_path)
    if not len(possible_hits) == 1:
        raise Exception('Cannot locate a unique data file given this information')
    else:
        filename = possible_hits[0]
        return filename


# ----------------------------------------------------------------------------------
def contrast_adjust_imgs(imgs_in, out_range=(0, 4095)):
    """
    Quick function to perform contrast adjustment on a set of video frames

    :param imgs_in: NxHxW array of video frames
    :param out_range: range to scale input pixel values to

    :return: imgs_out
    """
    from skimage import exposure

    # get input pixel range
    # vmin, vmax = np.percentile(imgs_in, q=(0.5, 99.5))
    vmin = np.min(imgs_in)
    vmax = np.max(imgs_in)

    # do rescaling
    imgs_out = exposure.rescale_intensity(imgs_in, in_range=(vmin, vmax),
                                          out_range=out_range)

    return imgs_out


# ----------------------------------------------------------------------------------
def my_read_cih(filename):
    """
    Pretty much the same as pyMRAW, just sans warnings/exceptions

    :param filename: full path to cih* file.

    :return: a dictionary containing metadata info (cih)

    """
    name, ext = os.path.splitext(filename)
    if ext == '.cih':
        cih = dict()
        # read the cif header
        with open(filename, 'r') as f:
            for line in f:
                if line == '\n':  # end of cif header
                    break
                line_sp = line.replace('\n', '').split(' : ')
                if len(line_sp) == 2:
                    key, value = line_sp
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                        cih[key] = value
                    except:
                        cih[key] = value

    elif ext == '.cihx':
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i]]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1] + 1])

        raw_cih_dict = xmltodict.parse(xml)
        cih = {
            'Date': raw_cih_dict['cih']['fileInfo']['date'],
            'Camera Type': raw_cih_dict['cih']['deviceInfo']['deviceName'],
            'Record Rate(fps)': float(raw_cih_dict['cih']['recordInfo']['recordRate']),
            'Shutter Speed(s)': float(raw_cih_dict['cih']['recordInfo']['shutterSpeed']),
            'Total Frame': int(raw_cih_dict['cih']['frameInfo']['totalFrame']),
            'Original Total Frame': int(raw_cih_dict['cih']['frameInfo']['recordedFrame']),
            'Image Width': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['width']),
            'Image Height': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['height']),
            'File Format': raw_cih_dict['cih']['imageFileInfo']['fileFormat'],
            'EffectiveBit Depth': int(raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['depth']),
            'EffectiveBit Side': raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['side'],
            'Color Bit': int(raw_cih_dict['cih']['imageDataInfo']['colorInfo']['bit']),
            'Comment Text': raw_cih_dict['cih']['basicInfo'].get('comment', ''),
        }

    else:
        raise Exception('Unsupported configuration file ({:s})!'.format(ext))

    # check exceptions
    ff = cih['File Format']
    if ff.lower() not in SUPPORTED_FILE_FORMATS:
        raise Exception('Unexpected File Format: {:g}.'.format(ff))
    bits = cih['Color Bit']
    if bits < 12:
        warnings.warn('Not 12bit ({:g} bits)! clipped values?'.format(bits))
        # - may cause overflow')
        # 12-bit values are spaced over the 16bit resolution - in case of photron filming at 12bit
        # this can be meanded by dividing images with //16
    if cih['EffectiveBit Depth'] != 12:
        warnings.warn('Not 12bit image!')
    # ebs = cih['EffectiveBit Side']
    # if ebs.lower() not in SUPPORTED_EFFECTIVE_BIT_SIDE:
    #     raise Exception('Unexpected EffectiveBit Side: {:g}'.format(ebs))
    # if (cih['File Format'].lower() == 'mraw') & (cih['Color Bit'] not in [8, 16]):
    #     raise Exception('pyMRAW only works for 8-bit and 16-bit files!')
    if cih['Original Total Frame'] > cih['Total Frame']:
        warnings.warn('Clipped footage! (Total frame: {}, Orig. total frame: {})'.format(cih['Total Frame'],
                                                                                         cih['Original Total Frame']))

    return cih


# ----------------------------------------------------------------------------------
def my_read_mraw(filename, raw_type='photron_mraw_mono_12bit', frames=None,
                 dtype=None):
    """
    This is really just a wrapper for pySciCam to read in images, but I'm hoping
    it streamlines video handling

    :param filename: full video file path; expects .MRAW
    :param raw_type: this is an input specifying file type for pySciCam. As long
                     as we're working with 12-bit images from PFV4, should stay
                     the same
    :param frames: tuple of indices with the form (start, end) that lets us load
                    a subset of images
    :param dtype: force the output images to have a data type e.g. np.uint8

    :return: images, metadata (numpy array containing video images and cih info)

    """
    # check input
    name, ext = os.path.splitext(filename)
    if not ext == '.mraw':
        raise Exception('Unexpected File Type: {:}'.format(ext))

    # get cih file info so we can input height and width to pySciCam
    cih_filename_search = glob.glob(name + '*.cih*')
    if not len(cih_filename_search) == 1:
        raise Exception('Could not locate associated CIH file for : {:}'.format(filename))
    cih_filename = cih_filename_search[0]

    metadata = my_read_cih(cih_filename)

    img_width = metadata['Image Width']
    img_height = metadata['Image Height']

    # call pySciCam to read images
    img_seq = ImageSequence(filename, rawtype=raw_type, width=img_width,
                            height=img_height, frames=frames, dtype=dtype)

    # return images and metadata
    return img_seq.arr, metadata


# ----------------------------------------------------------------------------------
def my_read_opencv(filename, frames=None):
    """
    Wrapper for reading in images with opencv
    (this is a little silly, but hopefully it will clean up future code)

    :param filename: video filename
    :param frames: tuple of indices with the form (start, end) that lets us load
                    a subset of images

    :return: images, metadata
    """
    # create video capture object
    cap = cv2.VideoCapture(filename)

    # grab metadata info
    name, ext = os.path.splitext(filename)

    # get cih file info so we can input height and width to pySciCam
    cih_filename_search = glob.glob(name + '*.cih*')
    if len(cih_filename_search) == 1:
        cih_filename = cih_filename_search[0]
        metadata = my_read_cih(cih_filename)

    elif len(cih_filename_search) > 1:
        # throw an exception if we find many cih files
        raise Exception('Detected multiple possible cih files for : {:}'.format(filename))

    else:
        # otherwise, if we don't have cih file, make an approximation
        import time
        metadata = {
            'Date': time.strftime('%Y/%m/%d', time.gmtime(os.path.getctime(filename))),
            'Camera Type': '',
            'Record Rate(fps)': float(cap.get(cv2.CAP_PROP_FPS)),
            'Shutter Speed(s)': None,
            'Total Frame': int(cv2.get(cv2.CAP_PROP_FRAME_COUNT)),
            'Original Total Frame': int(cv2.get(cv2.CAP_PROP_FRAME_COUNT)),
            'Image Width': int(cv2.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'Image Height': int(cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'File Format': ext,
            'EffectiveBit Depth': None,
            'EffectiveBit Side': None,
            'Color Bit': None,
            'Comment Text': None,
            }

    # get range of images to read
    if frames is None:
        start_idx = 0
        end_idx = int(cv2.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        start_idx = frames[0]
        end_idx = frames[-1]

    # read video frames and add to output
    images = list()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frame_counter = start_idx

    while frame_counter <= end_idx:
        # read current frame
        ret, frame = cap.read()

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # add to list
        images.append(gray)

        # increment counter
        frame_counter += 1

    # release video capture object
    cap.release()

    # convert images to numpy array
    images = np.concatenate(images)

    # return images and metadata
    return images, metadata


# ------------------------------------------------------------------------------
def save_merged_vid(filenames, filename_out, frames=None, fps=MERGED_FPS,
                    transform_list=None, display_flag=MERGED_DISPLAY_FLAG):
    """
    Function to take multiple video files, stitch them together, and save output

    :param filenames: list of filenames for videos to merge
    :param filename_out: where to save resultant combined video
    :param frames: tuple indicating which frame to start and stop at (optional)
    :param fps: frame rate for output video
    :param transform_list: list of same length as 'filenames' containing
                           transformations to be performed on images
    :param display_flag: boolean to indicate whether or not we should play video

    :return: [nothing]
    """
    # --------------------------------------------------------
    # read in images for video files
    imgs = []
    for fn in filenames:
        # how we load video file will depend on file type
        f, ext = os.path.splitext(fn)

        # if mraw format, use function from above
        if ext == '.mraw':
            imgs.append(my_read_mraw(fn, frames=frames)[0])

        # otherwise, assume it's something we can do in OpenCV
        else:
            imgs.append(my_read_opencv(fn, frames=frames)[0])

    # --------------------------------------------------------
    # perform any image transformations we might want
    # ... under construction. doing a temp thing now
    for jth, f in enumerate(filenames):
        if 'C001' in f:
            imgs[jth] = np.flip(imgs[jth], axis=1)

        elif 'C002' in f:
            imgs[jth] = contrast_adjust_imgs(imgs[jth])

    # --------------------------------------------------------
    # stitch images together
    imgs = np.concatenate(imgs, axis=2)

    # also convert to 8bit for opencv
    imgs = (imgs / 16).astype(np.uint8)

    # ---------------------------------------------
    # generate video writer object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc('H264')
    fps = int(fps)
    frame_size = (imgs.shape[2], imgs.shape[1])

    vid_writer = cv2.VideoWriter(filename_out, fourcc, fps,
                                 frame_size, isColor=False)

    # ----------------------------------------------
    # loop through frames and write video
    if display_flag:
        cv2.namedWindow('combined views', cv2.WINDOW_NORMAL)
        playback_dt = int((float(fps)/1000.)**(-1))

    for ith in range(imgs.shape[0]):

        # current image
        img_curr = np.squeeze(imgs[ith, :, :])

        # display images, if told to do so
        if display_flag:
            cv2.imshow('combined views', img_curr)

            # press ESC to exit
            k = cv2.waitKey(playback_dt) & 0xff
            if k == 27:
                break

        # write combined image to new video file
        vid_writer.write(img_curr)

    # release video writer
    vid_writer.release()

    print('finished writing new video: \n %s' %(filename_out))

    return
