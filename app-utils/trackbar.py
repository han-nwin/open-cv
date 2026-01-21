from __future__ import (
    print_function,
)  # Ensures compatibility of the print function between Python 2 and 3
from __future__ import (
    division,
)  # Ensures division operator behaves like Python 3 (true division) even in Python 2
import cv2 as cv
import argparse

alpha_slider_max = 100
title_window = "Linear Blend"


def on_trackbar(val):
    # NOTE: Alpha blend logic here
    alpha = val / alpha_slider_max
    beta = 1.0 - alpha
    dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    cv.imshow(title_window, dst)


parser = argparse.ArgumentParser(
    description="Code for Adding a Trackbar to our applications tutorial."
)
parser.add_argument(
    "--input1", help="Path to the first input image.", default="../asset/LinuxLogo.jpg"
)
parser.add_argument(
    "--input2",
    help="Path to the second input image.",
    default="../asset/WindowsLogo.jpg",
)
args = parser.parse_args()

# NOTE: First, we load two images, which are going to be blended
src1 = cv.imread(cv.samples.findFile(args.input1))
src2 = cv.imread(cv.samples.findFile(args.input2))

if src1 is None:
    print("Could not open or find the image: ", args.input1)
    exit(0)

if src2 is None:
    print("Could not open or find the image: ", args.input2)
    exit(0)

# NOTE: To create a trackbar, first we have to create the window
# in which it is going to be located
cv.namedWindow(title_window)

# Now we can create the Trackbar:
trackbar_name = "Alpha x %d" % alpha_slider_max
cv.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)

# Show some stuff
# with the callback function
on_trackbar(0)

# Wait until user press some key
cv.waitKey()
