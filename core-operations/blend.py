import cv2 as cv

img1 = cv.imread("../asset/ml.png")
img2 = cv.imread("../asset/opencv-logo.png")
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
#
# Resize img2 to match img1
img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()
