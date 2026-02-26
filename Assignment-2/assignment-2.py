import numpy as np
import cv2
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(working_dir, "lena_512.jpg"), cv2.IMREAD_GRAYSCALE)


# Manual convolution
def apply_filter(image, kernel):
    kh, kw = kernel.shape  # get kernel size
    # Pad image to avoid edge artifacts when convolving with kernel
    pad_h, pad_w = kh // 2, kw // 2  # get padding
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect").astype(
        np.float64
    )  # pad image with reflected pixels mode
    output = np.zeros_like(image, dtype=np.float64)  # create output array

    # Convolve image on each axis pixel by pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


# --- a. 7x7 Box Blur Filter ---
box_kernel = np.ones((7, 7), dtype=np.float64) / 49.0
box_result = apply_filter(img, box_kernel)
cv2.imwrite(os.path.join(working_dir, "a_box_blur_7x7.jpg"), to_uint8(box_result))
print("Saved a_box_blur_7x7.jpg")

# --- b. 15x15 Gaussian Filter ---
kernel1d = cv2.getGaussianKernel(15, 4.0)
kernel2d = np.outer(kernel1d, kernel1d)
gaussian_result = apply_filter(img, kernel2d)
cv2.imwrite(
    os.path.join(working_dir, "b_gaussian_15x15.jpg"), to_uint8(gaussian_result)
)
print("Saved b_gaussian_15x15.jpg")

# --- c. 15x15 Motion Blur Filter ---
kernel = np.zeros((15, 15))
np.fill_diagonal(kernel, 1)
kernel = kernel / 15
motion_result = apply_filter(img, kernel)
cv2.imwrite(
    os.path.join(working_dir, "c_motion_blur_15x15.jpg"), to_uint8(motion_result)
)
print("Saved c_motion_blur_15x15.jpg")

# --- d. 3x3 Laplacian Sharpening Filter ---
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float64)
laplacian_result = apply_filter(img, kernel)
cv2.imwrite(
    os.path.join(working_dir, "d_laplacian_sharpen_3x3.jpg"), to_uint8(laplacian_result)
)
print("Saved d_laplacian_sharpen_3x3.jpg")

# --- e. Canny Edge Detection ---
# i. 5x5 Gaussian Filter to smooth the image
K = (1.0 / 159) * np.array(
    [
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2],
    ],
    dtype=np.float64,
)
smoothed = apply_filter(img, K)  # de-noised image
cv2.imwrite(os.path.join(working_dir, "e-i_gaussian_5x5.jpg"), to_uint8(smoothed))
print("Saved e-i_gaussian_5x5.jpg")

# ii. Sobel Kernels
Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

sobel_x = apply_filter(smoothed, Gx)
sobel_y = apply_filter(smoothed, Gy)

# Calculate magnitude and direction
M = np.sqrt(sobel_x**2 + sobel_y**2)
theta = np.arctan2(sobel_y, sobel_x)

cv2.imwrite(os.path.join(working_dir, "e-ii_sobel.jpg"), to_uint8(M))
print("Saved e-ii_sobel.jpg")

# iii. Non-Maximum Suppression
rows, cols = M.shape
nms = np.zeros_like(M)

# Convert angle to degrees and map to 0-180
angle = np.degrees(theta) % 180

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        a = angle[i, j]

        # Determine two neighbors to compare along gradient direction
        if (0 <= a < 22.5) or (157.5 <= a <= 180):
            n1, n2 = M[i, j - 1], M[i, j + 1]
        elif 22.5 <= a < 67.5:
            n1, n2 = M[i + 1, j - 1], M[i - 1, j + 1]
        elif 67.5 <= a < 112.5:
            n1, n2 = M[i - 1, j], M[i + 1, j]
        else:
            n1, n2 = M[i - 1, j - 1], M[i + 1, j + 1]

        # Keep pixel only if it's a local maximum
        if M[i, j] >= n1 and M[i, j] >= n2:
            nms[i, j] = M[i, j]

cv2.imwrite(os.path.join(working_dir, "e-iii_nms.jpg"), to_uint8(nms))
print("Saved e-iii_nms.jpg")

# iv. Thresholding
high_thresh = 0.15 * np.max(nms)
low_thresh = 0.05 * np.max(nms)

strong = 255
weak = 50

result = np.zeros_like(nms)
result[nms >= high_thresh] = strong
result[(nms >= low_thresh) & (nms < high_thresh)] = weak

# Hysteresis: keep weak edges only if connected to a strong edge
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        if result[i, j] == weak:
            if np.any(result[i - 1 : i + 2, j - 1 : j + 2] == strong):
                result[i, j] = strong
            else:
                result[i, j] = 0

cv2.imwrite(os.path.join(working_dir, "e-iv_canny.jpg"), result.astype(np.uint8))
print("Saved e-iv_canny.jpg")
