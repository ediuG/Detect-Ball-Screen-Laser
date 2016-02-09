import numpy as np
import cv2
import cv2.cv as cv
import sys
import time
from common import draw_str

__author__ = 'Guide-ce'

class ScreenPosition(object):

    """docstring for ScreenPosition"""

    def __init__(self, cam_width=1024, cam_height=768):
        super(ScreenPosition, self).__init__()
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.top_left = {}
        self.bottom_right = {}

        self.capture = None
        self.success = False

    def setup_camera_capture(self, device_num=1):
        """Perform camera setup for the device number (default device = 0).
        Returns a reference to the camera Capture object.

        """
        try:
            device = int(device_num)
            sys.stdout.write("Using Camera Device: {0}\n".format(device))
        except (IndexError, ValueError):
            # assume we want the 1st device
            device = 0
            sys.stderr.write("Invalid Device. Using default device 0\n")

        # Try to start capturing frames
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            sys.stderr.write("False to Open Capture device. Quitting.\n")
            sys.exit(1)

        # set the wanted image size from the camera
        self.capture.set(
            cv.CV_CAP_PROP_FRAME_WIDTH,
            self.cam_width
        )
        self.capture.set(
            cv.CV_CAP_PROP_FRAME_HEIGHT,
            self.cam_height
        )
        return self.capture

    def image_preparation(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grey_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, thre = cv2.threshold(grey_image, 127, 255, cv.CV_THRESH_BINARY)

        return thre

    def detect(self, image):
        contours, hierarchy = cv2.findContours(image, cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)
        for h,cnt in enumerate(contours):
			approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        # moments = cv2.moments(cnt)
        # x,y,w,h = cv2.boundingRect(contours)
        # cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        # print moments

    def handle_success(self):
        pass


    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            sys.exit(0)

    def run(self):
        self.setup_camera_capture()

        while True:
            ret, frame = self.capture.read()
            if not ret:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)

            thre_image = self.image_preparation(frame)
            self.detect(thre_image)
            if self.handle_success() is True:
                break
            self.handle_quit()

if __name__ == '__main__':
    screen = ScreenPosition()
    screen.run()
