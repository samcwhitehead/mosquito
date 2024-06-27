"""
Scratch file for general video viewer -- want to eventually use this sort of thing
both for high speed video data (.mraw) and tendon/muscle images
(or at least create a general enough framework that with minimal changes we
could use this code for both instances)

Author: Samuel C Whitehead

NB: run " designer " in terminal to open Qt designer
"""
# -----------------------------------------------------------------------------
# Imports
import sys
import cv2
import h5py
import pyqtgraph

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QFileDialog, QSlider, QHBoxLayout, QWidget, QVBoxLayout,
                             QPushButton)


# -----------------------------------------------------------------------------
# Define VideoViewer class
class VideoViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI file
        loadUi("video_viewer.ui", self)

        # Initialize the video file path, frame index, video frame count, etc
        self.file_path = None
        self.frame_idx = 0
        self.n_frames = 0

        # Connect frame slider signal to slot
        self.frameSlider.valueChanged.connect(self.frame_slider_value_changed)

        # Connect play and pause button signals to slots
        self.playButton.clicked.connect(self.play_video)
        self.pauseButton.clicked.connect(self.pause_video)
        self.stopButton.clicked.connect(self.stop_video)

        # Connect frame skip button signals to slots
        self.ffwdButton.clicked.connect(self.frame_ffwd)
        self.fwdButton.clicked.connect(self.frame_fwd)
        self.backButton.clicked.connect(self.frame_back)
        self.bbackButton.clicked.connect(self.frame_bback)

        # Connect loop video checkbox signal to slot
        self.loopCheckBox.stateChanged.connect(self.toggle_video_loop)

        # Connect frames per second (playback) line edit signal to slot

    # ----------------------------------------------------------------
    # Frame slider methods
    def frame_slider_value_changed(self, value):
        """ update frame_idx according to changes in frame slider """
        self.frame_idx = value
        self.update_frame_display()

    def update_frame_slider_value(self):
        """ update slider position based on current frame """
        self.frameSlider.setValue(self.frame_idx)

    # ----------------------------------------------------------------
    # Video play/pause methods
    def play_video(self):
        pass

    def pause_video(self):
        pass

    def stop_video(self):
        pass

    def toggle_video_loop(self, state):
        """ checkbox for whether or not to loop video continuously """
        if state == Qt.Checked:
            print('Loop video (under construction)')
        else:
            print('DO NOT loop video (under construction)')

    def set_playback_fps(self, value):
        pass

    # ----------------------------------------------------------------
    # Frame skip methods
    def frame_ffwd(self):
        """ skip to final frame """
        # Under construction!
        self.update_frame_display()
        self.update_frame_slider_value()

    def frame_fwd(self):
        """ skip forward one frame """
        self.frame_idx += 1
        self.update_frame_display()
        self.update_frame_slider_value()

    def frame_back(self):
        """ skip backward one frame """
        self.frame_idx -= 1
        self.update_frame_display()
        self.update_frame_slider_value()

    def frame_bback(self):
        """ skip to first frame """
        self.frame_idx = 0
        self.update_frame_display()
        self.update_frame_slider_value()

    # ----------------------------------------------------------------
    # Display update methods
    def update_frame_display(self):
        """ update the displayed video frame """
        print(self.frame_idx)


# -----------------------------------------------------------------------------
# What to run at main
def main():
    app = QApplication(sys.argv)
    main_gui = VideoViewer()
    main_gui.show()
    sys.exit(app.exec_())


# -----------------------------------------------------------------------------
# Main script
if __name__ == '__main__':
   main()
