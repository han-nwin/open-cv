import numpy as np
import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(script_dir, "lena_512.jpg"), cv2.IMREAD_GRAYSCALE)


# Manual convolution
def apply_filter(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect").astype(
        np.float64
    )
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


# --- a. 7x7 Box Blur Filter ---
box_kernel = np.ones((7, 7), dtype=np.float64) / 49.0
box_result = apply_filter(img, box_kernel)
cv2.imwrite(os.path.join(script_dir, "box_blur_7x7.jpg"), box_result)
print("Saved box_blur_7x7.jpg")

# --- b. 15x15 Gaussian Filter ---
kernel1d = cv2.getGaussianKernel(15, 4.0)
gaussian_kernel = np.outer(kernel1d, kernel1d)
gaussian_result = apply_filter(img, gaussian_kernel)
cv2.imwrite(os.path.join(script_dir, "gaussian_15x15.jpg"), gaussian_result)
print("Saved gaussian_15x15.jpg")
