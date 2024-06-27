"""
Code from ChatGPT that purports to allow users to load and play video files

"""
import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QFileDialog, QSlider, QHBoxLayout, QWidget, QVBoxLayout,
                            QPushButton)


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # # Create the video player widget
        # self.video_widget = QLabel(self)
        # self.video_widget.setGeometry(0, 0, int(self.width()), int(self.height() * 0.8))
        # self.video_widget.setAlignment(Qt.AlignCenter)
        #
        # # Create the slider bar widget for frame incrementation
        # self.slider_bar = QSlider(Qt.Horizontal, self)
        # self.slider_bar.setMinimum(0)
        # self.slider_bar.setMaximum(100)
        # self.slider_bar.setValue(0)
        # self.slider_bar.setTickPosition(QSlider.TicksBelow)
        # self.slider_bar.valueChanged.connect(self.slider_value_changed)
        #
        # # Create a horizontal box layout to hold the video widget and slider bar widget
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.video_widget)
        # hbox.addWidget(self.slider_bar)
        #
        # # Create a vertical box layout to hold the contrast and brightness sliders, and histogram equalization button
        # vbox = QVBoxLayout()
        #
        # # Create the contrast slider
        # self.contrast_slider = QSlider(Qt.Horizontal, self)
        # self.contrast_slider.setMinimum(-100)
        # self.contrast_slider.setMaximum(100)
        # self.contrast_slider.setValue(0)
        # self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        # self.contrast_slider.valueChanged.connect(self.contrast_value_changed)
        # vbox.addWidget(QLabel('Contrast'))
        # vbox.addWidget(self.contrast_slider)
        #
        # # Create the brightness slider
        # self.brightness_slider = QSlider(Qt.Horizontal, self)
        # self.brightness_slider.setMinimum(-100)
        # self.brightness_slider.setMaximum(100)
        # self.brightness_slider.setValue(0)
        # self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        # self.brightness_slider.valueChanged.connect(self.brightness_value_changed)
        # vbox.addWidget(QLabel('Brightness'))
        # vbox.addWidget(self.brightness_slider)
        #
        # # Create the histogram equalization button
        # self.equalize_button = QPushButton('Histogram Equalization', self)
        # self.equalize_button.clicked.connect(self.equalize_histogram)
        # vbox.addWidget(self.equalize_button)
        #
        # # Create a widget to hold the horizontal box layout and vertical box layout
        # widget = QWidget()
        # widget.setLayout(hbox)
        # vbox_widget = QWidget()
        # vbox_widget.setLayout(vbox)
        # layout = QVBoxLayout()
        # layout.addWidget(widget)
        # layout.addWidget(vbox_widget)
        # main_widget = QWidget()
        # main_widget.setLayout(layout)
        # self.setCentralWidget(main_widget)

        # Create label to display video frame
        self.video_widget = QLabel(self)
        self.video_widget.setGeometry(0, 0, int(self.width()), int(self.height() * 0.8))

        # Create the slider bar widget for frame incrementation
        self.slider_bar = QSlider(Qt.Horizontal, self)
        self.slider_bar.setMinimum(0)
        self.slider_bar.setMaximum(100)
        self.slider_bar.setValue(0)
        self.slider_bar.setGeometry(
            10, int(self.height() * 0.80), int(self.width() * 0.8), 20)
        self.slider_bar.setTickPosition(QSlider.TicksBelow)
        self.slider_bar.valueChanged.connect(self.slider_value_changed)

        # Create slider to adjust contrast
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.setGeometry(
            10, int(self.height() * 0.85), int(self.width() * 0.8), 20)
        self.contrast_slider.valueChanged.connect(self.contrast_value_changed)

        # Create slider to adjust brightness
        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.setGeometry(
            10, int(self.height() * 0.9), int(self.width() * 0.8), 20)
        self.brightness_slider.valueChanged.connect(self.brightness_value_changed)

        # Create button to play video
        self.play_button = QPushButton("Play", self)
        self.play_button.setGeometry(
            int(self.width() * 0.85), int(self.height() * 0.85), 60, 30)
        self.play_button.clicked.connect(self.play_video)

        # Create button to pause video
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setGeometry(
            int(self.width() * 0.85), int(self.height() * 0.9), 60, 30)
        self.pause_button.clicked.connect(self.pause_video)

        # Create button to stop video
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(
            int(self.width() * 0.85), int(self.height() * 0.95), 60, 30)
        self.stop_button.clicked.connect(self.stop_video)

        # Create the menu bar
        menu_bar = self.menuBar()

        # Add the "File" menu
        file_menu = menu_bar.addMenu("File")

        # Add the "Open" action to the "File" menu
        open_action = file_menu.addAction("Open")
        open_action.triggered.connect(self.open_video_file)

        # Set the window properties
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # # Create the timer object for frame display
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.display_frame)

        # Initialize the video file path and frame index
        self.video_file_path = None
        self.frame_index = 0
        self.num_frames = 0
        self.current_frame = None

    def open_video_file(self):
        # Prompt the user to select a video file
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mov)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            self.video_file_path = file_dialog.selectedFiles()[0]
            self.frame_index = 0

            # Start the video player
            self.start_video_player()

    def start_video_player(self):
        # Open the video file
        self.video_capture = cv2.VideoCapture(self.video_file_path)
        self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the maximum value of the slider bar to the number of frames in the video
        self.slider_bar.setMaximum(self.num_frames)

        # # Start the timer for frame display
        # self.timer.start(int(1000 // self.video_capture.get(cv2.CAP_PROP_FPS)))

        # Display initial frame
        self.display_frame()

    def display_frame(self):
        # Read the next frame from the video file
        ret, frame = self.video_capture.read()

        if ret:
            # Convert the frame to a QImage
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_BGR888)

            # Scale the image to fit the video widget
            scaled_image = image.scaled(self.video_widget.size(), Qt.KeepAspectRatio)

            # Set the image to the video widget
            self.video_widget.setPixmap(QPixmap.fromImage(scaled_image))

            # Increment the frame index
            self.frame_index += 1

            # Update the value of the slider bar
            self.slider_bar.setValue(self.frame_index)

    def slider_value_changed(self, value):
        self.frame_index = value  # int(value / 100 * self.num_frames)
        print(self.frame_index)

        # update video capture frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index - 1)

        # update display
        self.display_frame()

    def contrast_value_changed(self, value):
        """
        The value parameter is the current value of the contrast slider, which ranges from 0 to 100.
        We scale it to the range -1 to 1 by subtracting 50 and dividing by 50.0.

        We then create a copy of the original frame (self.original_frame) and apply the contrast
        adjustment using OpenCV's cv2.addWeighted function. The cv2.addWeighted function adds two
        images with adjustable weights, and in this case we use it to adjust the contrast of the image.

        Finally, we call the display_frame() method to update the displayed frame with the
        contrast-adjusted version.

        """
        contrast = (value - 50) / 50.0
        self.current_frame = self.original_frame.copy()
        if contrast != 0:
            self.current_frame = cv2.addWeighted(
                self.current_frame, contrast + 1,
                np.zeros(self.current_frame.shape, self.current_frame.dtype), 0,
                -128 * contrast)
        self.display_frame()

    def brightness_value_changed(self, value):
        """
        The value parameter is the current value of the brightness slider, which ranges from 0 to 100.
        We scale it to the range -255 to 255 by subtracting 50, dividing by 50.0, and multiplying by 255.

        We then create a copy of the original frame (self.original_frame) and add the brightness
        adjustment to it using NumPy's np.clip function to ensure that the pixel values stay within the
        range 0 to 255.

        Finally, we call the display_frame() method to update the displayed frame with the
        brightness-adjusted version.
        """
        brightness = (value - 50) / 50.0 * 255
        self.current_frame = self.original_frame.copy()
        self.current_frame = np.clip(self.current_frame + brightness, 0, 255)
        self.display_frame()

    def equalize_histogram(self):
        """
        In this method, we first create a copy of the original frame (self.original_frame).
        We then convert it to grayscale using OpenCV's cv2.cvtColor function, which takes the
        BGR image as input and converts it to grayscale.

        Next, we apply adaptive histogram equalization using OpenCV's cv2.equalizeHist function.
        This function applies contrast-limited adaptive histogram equalization to the grayscale
        image, which improves the contrast and enhances details in the image.

        Finally, we convert the grayscale image back to BGR using cv2.cvtColor and call the
         display_frame() method to update the displayed frame with the equalized version.
        """
        self.current_frame = self.original_frame.copy()
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        self.current_frame = cv2.equalizeHist(self.current_frame)
        self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        self.display_frame()

    def play_video(self):
        pass

    def stop_video(self):
        pass

    def pause_video(self):
        pass


# -----------------------------------------------------------------------------
# Main script
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec_())
