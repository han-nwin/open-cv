"""
Testing different preprocessing approaches to see what works and what doesn't.
"""
import os
import cv2
import numpy as np


def preprocess_v1_bad(img_path):
    """Bad: simple threshold, no blur."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    # Fixed threshold, no blur
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    normalized = binary.astype(np.float32) / 255.0
    return normalized


def preprocess_v2_no_blur(img_path):
    """Bad: adaptive threshold but no blur."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    # Adaptive threshold but skipping blur
    binary = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    normalized = binary.astype(np.float32) / 255.0
    return normalized


def preprocess_v3_good(img_path):
    """Good: blur + adaptive threshold (what we actually use)."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    normalized = binary.astype(np.float32) / 255.0
    return normalized


if __name__ == "__main__":
    # Test on all images
    test_images = []
    for d in ["dataset/train/own", "dataset/test/own", "dataset/train/others", "dataset/test/others"]:
        files = sorted(os.listdir(d))
        for f in files:
            test_images.append(os.path.join(d, f))

    for img_path in test_images:
        print(f"\n{img_path}:")

        v1 = preprocess_v1_bad(img_path)
        v2 = preprocess_v2_no_blur(img_path)
        v3 = preprocess_v3_good(img_path)

        # Save side by side comparison
        basename = os.path.basename(img_path)
        cv2.imwrite(f"preprocess_v1_{basename}", (v1 * 255).astype(np.uint8))
        cv2.imwrite(f"preprocess_v2_{basename}", (v2 * 255).astype(np.uint8))
        cv2.imwrite(f"preprocess_v3_{basename}", (v3 * 255).astype(np.uint8))
        print(f"  v1 (fixed threshold, no blur): saved")
        print(f"  v2 (adaptive, no blur): saved")
        print(f"  v3 (blur + adaptive): saved")
