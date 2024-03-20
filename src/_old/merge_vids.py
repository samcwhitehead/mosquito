"""
Quick script to merge videos from different views together and save them as a single file

"""
import os
import glob
import cv2

import numpy as np


#########################################################################################
##################### PARAMS/DATA INFO ##################################################
#########################################################################################
DATA_PATH = "/media/sam/SamData/HighSpeedVideo/SamHingeTwoCameraTest"
CAMERAS = ["Camera_1", "Camera_2"]
DATE_STR = "20221122_171306"
FILE_EXT = ".avi"
SAVE_STR = "combined"
DISPLAY_FLAG = True
FPS = 50


#########################################################################################
##################### MERGE FUNCTION ####################################################
#########################################################################################

def merge_video_files(date_str=DATE_STR, data_path=DATA_PATH, cameras=CAMERAS, fps=FPS,
                      save_str=SAVE_STR, file_ext_in=FILE_EXT, file_ext_out=FILE_EXT,
                      display_flag=DISPLAY_FLAG):
    """
    Function to locate data files, load them in, and create a single merged output file
    :param date_str: string in the format 'YYYYMMDD_HHMMSS' that identifies movie file
    :param data_path: folder containing video files
    :param cameras: list of strings giving camera names (these are assumed to be in
                    the filenames of the videos)
    :param fps: frame rate for output video
    :param save_str: string used as prefix for output file/folder
    :param file_ext_in: extension for input files
    :param file_ext_out: extension for output file
    :param display_flag: boolean to indicate whether or not we should play video

    :return: Nothing at the moment
    """
    # --------------------------------------------------------
    # locate paths to the video files we care about

    folder_search_str = os.path.join(data_path, "%s_*_%s", "*", "*%s")
    video_paths = [glob.glob(folder_search_str % (cam, date_str, file_ext_in))[0] for cam in cameras]

    # do a little print check
    [print(p + "\n") for p in video_paths]

    # ----------------------------------
    # get folder to save output to
    save_path = os.path.join(data_path, '%s_%s'%(save_str, date_str))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_fn = '%s_%s%s'%(save_str, date_str, file_ext_out)
    save_path_full = os.path.join((save_path, save_fn))

    # --------------------------------------------------------
    # generate video capture objects to read in video frames
    vid_caps = [cv2.VideoCapture(path) for path in video_paths]

    # also get dimensions for each video
    vid_widths = [cap.get(3) for cap in vid_caps]
    vid_heights = [cap.get(4) for cap in vid_caps]

    # ---------------------------------------------
    # generate video writer object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = int(fps)
    frame_size = (sum(vid_widths), max(vid_heights))

    vid_writer = cv2.VideoWriter(save_path_full, fourcc, fps, frame_size)

    # ----------------------------------------------
    # loop through frames and write video
    if display_flag:
        cv2.namedWindow('combined views', cv2.WINDOW_NORMAL)
        playback_dt = (float(fps)/1000.)**(-1)

    while all([cap.isOpened() for cap in vid_caps]):
        # read both frames and boolean return values
        rets, frames = list(zip(*[cap.read() for cap in vid_caps]))

        # make sure we got frames
        if not all(rets):
            print("Can't read frames -- exiting")
            break

        # if we read images, combine them (don't think I need to convert)
        # imgs = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        frame_combined = np.hstack(tuple(frames))

        # display images, if told to do so
        if display_flag:
            cv2.imshow('combined views', frame_combined)

            # press ESC to exit
            k = cv2.waitKey(playback_dt) & 0xff
            if k == 27:
                break

        # write combined image to new video file
        vid_writer.write(frame_combined)

    # release video captures and writers
    [cap.release() for cap in vid_caps]
    vid_writer.release()
    print('finished writing new video: \n %s' %(save_path_full))

    return


#########################################################################################
########################## MAIN #########################################################
#########################################################################################

if __name__ == '__main__':
    merge_video_files()
