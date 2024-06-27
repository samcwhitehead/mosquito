"""
Quick script to test combining mraw videos into merged avi for viewing

"""
import os
from src.read_photron import get_vid_filename, save_merged_vid

#########################################################################################
######################## PARAMS/CONSTANTS ###############################################
#########################################################################################
ROOT_PATH = '/media/sam/SamData/HighSpeedVideo'
EXPT_FOLDER = '01_13012023'

CAM_NUMS = [1, 2]
HEADER_NUM = 1
PART_NUM = 3
DATE_STR = None

SEARCH_FILE_EXT = '.mraw'
FILE_ID_FORMAT = 'C%03dH%03dS%04d'

FRAMES = (0, 1500)
DISPLAY_FLAG = True
FPS = 50
OUTPUT_EXT = '.avi'
OUTPUT_CODEC = ''

#########################################################################################
############################# MAIN SCRIPT ###############################################
#########################################################################################
if __name__ == '__main__':
    """ Run the script to merge/save videos """
    # get data filenames
    filenames = list()
    for cnum in CAM_NUMS:
        fn = get_vid_filename(root_path=ROOT_PATH, expt_folder=EXPT_FOLDER,
                              sub_folder=None, cam_num=cnum, header_num=HEADER_NUM,
                              partition_num=PART_NUM, file_ext=SEARCH_FILE_EXT,
                              file_id_format=FILE_ID_FORMAT, date_str=DATE_STR)
        filenames.append(fn)

    # output filename
    path = os.path.dirname(filenames[0])
    parent_dir = os.path.abspath(os.path.join(path, '..'))

    save_str = 'combined_' + FILE_ID_FORMAT %(0, HEADER_NUM, PART_NUM)
    save_dir = os.path.join(parent_dir, save_str)
    save_fn = save_str
    if DATE_STR is None:
        pass

    else:
        save_dir += ('_' + DATE_STR)
        save_fn += ('_' + DATE_STR)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    filename_out = os.path.join(save_dir, save_fn + OUTPUT_EXT)

    # do merge/save
    save_merged_vid(filenames, filename_out, frames=FRAMES, fps=FPS,
                    transform_list=None, display_flag=DISPLAY_FLAG)
