from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

# Read image given by user

parser = argparse.ArgumentParser(
    description="Code for Changing the contrast and "
    "brightness of an image! tutorial."
)
parser.add_argument("--input", help="Path to input image.",
                    default="../asset/lena.jpg")
args = parser.parse_args()

image = cv.imread(cv.samples.findFile(args.input))
if image is None:
    print("Could not open or find the image: ", args.input)
    exit(0)

new_image = np.zeros(image.shape, image.dtype)

alpha_slider_max = 10
beta_slider_max = 100
title_window = "Linear Transforms"

alpha = 0
beta = 0

cv.namedWindow(title_window)
# Alpha trackbar
trackbar_name_alpha = "Alpha x %d" % alpha_slider_max
# Beta trackbar
trackbar_name_beta = "Beta x %d" % beta_slider_max


def on_trackbar(val):
    alpha = cv.getTrackbarPos(trackbar_name_alpha, title_window) + 1
    beta = cv.getTrackbarPos(trackbar_name_beta, title_window)
    new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    cv.imshow(title_window, new_image)


cv.createTrackbar(trackbar_name_alpha, title_window,
                  0, alpha_slider_max, on_trackbar)
cv.createTrackbar(trackbar_name_beta, title_window,
                  0, beta_slider_max, on_trackbar)


on_trackbar(0)

cv.waitKey()
cv.destroyAllWindows()
