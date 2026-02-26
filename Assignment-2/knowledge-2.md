# Assignment 2 Filters Explained

## a. 7x7 Box Blur

The simplest blur. Every pixel in the 7x7 neighborhood gets equal weight (1/49 each).

**How it works:** For each pixel, take the average of all 49 pixels in its 7x7 neighborhood. This smooths out differences between neighboring pixels, making the image blurry.

**Why it looks blocky:** Since every neighbor is weighted equally, even far-away pixels in the 7x7 window contribute the same as immediate neighbors. This can create visible square artifacts compared to Gaussian blur.

---

## b. 15x15 Gaussian Blur

Like box blur but smarter — center pixels get more weight, far pixels get less. The weights follow a bell curve (Gaussian distribution).

**How it works:** The kernel is generated from a 1D Gaussian curve (controlled by sigma=4.0), then made 2D with `np.outer`. Pixels closer to the center contribute more to the average, pixels farther away contribute less.

**Why sigma matters:** Sigma controls the width of the bell curve. Larger sigma = wider spread = more blur. Sigma=4.0 with a 15x15 kernel gives a moderate blur.

**Why it looks better than box blur:** The smooth falloff from center to edges produces a more natural-looking blur without the blocky artifacts.

---

## c. 15x15 Motion Blur

Simulates camera movement during exposure by averaging pixels along a diagonal line.

**How it works:** The kernel is all zeros except for 1s on the main diagonal, normalized by dividing by 15. This means each output pixel is the average of 15 pixels along a diagonal streak.

```
| 1  0  0  ... |
| 0  1  0  ... |
| 0  0  1  ... |   / 15
| ...       ... |
```

**Why diagonal:** The 1s sit on the top-left to bottom-right diagonal, so the blur goes in that direction. Changing which entries are 1 would change the blur direction (e.g., a horizontal row of 1s would give horizontal motion blur).

---

## d. 3x3 Laplacian Sharpening

Makes edges in the image more pronounced.

**How it works:** The kernel has 9 in the center and -1 everywhere around it:

```
| -1  -1  -1 |
| -1   9  -1 |
| -1  -1  -1 |
```

This is really doing: `original pixel + (original pixel - average of neighbors)`. The center weight of 9 = 1 (keep original) + 8 (boost the difference from neighbors). Where pixel values change sharply (edges), the difference is large, so edges get amplified. In smooth areas, the pixel roughly equals its neighbors, so little changes.

**Why values sum to 1:** (-1×8) + 9 = 1. This preserves overall brightness. If it summed to 0, you'd only get edges on a black background. Summing to 1 means you get the original image with enhanced edges.

---

## e. Canny Edge Detection

A multi-step pipeline to find clean, thin edges.

### e.i. 5x5 Gaussian Smoothing

Blur the image first to remove noise. Without this, the edge detector would pick up tiny noise speckles as "edges."

The specific kernel given is a discrete approximation of a Gaussian with all values summing to 159 (hence the 1/159 normalization).

### e.ii. Sobel Kernels

Two separate 3x3 kernels that detect edges in different directions:

- **Gx** detects vertical edges (pixel intensity changes left-to-right)
- **Gy** detects horizontal edges (pixel intensity changes top-to-bottom)

Apply both to the smoothed image, then combine:
- **Magnitude** `M = sqrt(Gx² + Gy²)` — how strong is the edge at this pixel?
- **Direction** `theta = arctan(Gy/Gx)` — which way does the edge point?

### e.iii. Non-Maximum Suppression

After Sobel, edges are thick/blurry blobs. This step thins them to 1 pixel wide.

For each pixel, look at its two neighbors along the gradient direction. If this pixel isn't the strongest of the three, set it to 0. Only the local maximum along each edge direction survives.

### e.iv. Thresholding

Clean up by classifying pixels into three categories:
- **Strong edges:** magnitude > high threshold — definitely an edge, keep it
- **Weak edges:** between low and high threshold — only keep if connected to a strong edge
- **Not edges:** below low threshold — set to 0

The "connected to a strong edge" rule (hysteresis) prevents broken edge lines while still filtering out noise.
