# Image Filtering - Kernels & Convolution Explained

## What is a Kernel?

A kernel is just a small grid of numbers (a matrix). For example, a 3x3 kernel:

```
| 1  1  1 |
| 1  1  1 |
| 1  1  1 |
```

That's it. It's just numbers in a grid. The size can vary — 3x3, 5x5, 7x7, 15x15, etc.

---

## What is Convolution? (How we "apply" a kernel)

Convolution is the process of sliding the kernel over every pixel in the image and computing a new value for that pixel.

### Step by step:

Say we have this 5x5 image and a 3x3 kernel:

```
Image:                  Kernel:
| 10  20  30  40  50 |  | 1  1  1 |
| 60  70  80  90 100 |  | 1  1  1 |  * (1/9)
|110 120 130 140 150 |  | 1  1  1 |
|160 170 180 190 200 |
|210 220 230 240 250 |
```

To find the new value of the CENTER pixel (130):

1. Place the kernel centered on pixel 130
2. The kernel covers this 3x3 region:

```
|  70   80   90 |
| 120  130  140 |
| 170  180  190 |
```

3. Multiply each value by the corresponding kernel value:

```
70*1 + 80*1 + 90*1 + 120*1 + 130*1 + 140*1 + 170*1 + 180*1 + 190*1 = 1170
```

4. Multiply by the normalizing factor: `1170 * (1/9) = 130`

5. That's the new pixel value! Then slide the kernel to the next pixel and repeat.

---

## What About Edge Pixels? (Padding)

When the kernel is centered on an edge/corner pixel, part of the kernel hangs outside the image — those pixels don't exist.

Example: kernel centered on the top-left corner pixel:

```
|  ?   ?   ?  |
|  ?  [10]  20 |
|  ?   60   70 |
```

The `?` values don't exist. We need to **pad** the image — fill in those missing values before convolving. Common methods:

### 1. Zero Padding — fill missing values with 0

```
|  0    0   0  |
|  0  [10]  20 |
|  0   60   70 |
```

Downside: edge pixels get darker because you're averaging with 0s.

### 2. Reflect Padding — mirror the image at the border

```
|  70   60   70 |
|  20  [10]  20 |
|  70   60   70 |
```

This looks the most natural. **Most commonly used in practice.**

### 3. Replicate/Clamp — repeat the nearest edge pixel

```
|  10   10   20 |
|  10  [10]  20 |
|  60   60   70 |
```

### Which one should you use?

- **Reflect** is the most common default in practice and libraries
- **Zero padding** is simpler to implement and common in academic settings
- **Check your class slides** — your professor may expect a specific method. The assignment doesn't specify, so either zero or reflect should be fine, but confirm with your slides to be safe

In our code, this line handles it:

```python
padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
```

This adds extra rows/columns around the image before we start convolving. By the time the kernel reaches an edge pixel, those padded values are already there — the kernel never sees missing data.

---

## The Filters in This Assignment

### a. Box Blur (7x7)

Every value in the kernel is the same: `1/49`

```
| 1  1  1  1  1  1  1 |
| 1  1  1  1  1  1  1 |
| 1  1  1  1  1  1  1 |
| 1  1  1  1  1  1  1 |  * (1/49)
| 1  1  1  1  1  1  1 |
| 1  1  1  1  1  1  1 |
| 1  1  1  1  1  1  1 |
```

**What it does:** Replaces each pixel with the average of its 7x7 neighborhood. Everything is weighted equally, so it blurs the image uniformly.

**Why 1/49?** A 7x7 kernel has 49 values. Dividing by 49 makes them sum to 1, so the overall brightness stays the same.

---

### b. Gaussian Blur (15x15)

Similar to box blur, but the center pixels have MORE weight and edge pixels have LESS weight. It follows a bell curve (Gaussian distribution).

```
        Low   Low   Low
  Low   Med   Med   Med   Low
  Low   Med   HIGH  Med   Low
  Low   Med   Med   Med   Low
        Low   Low   Low
```

**What it does:** Blurs the image, but more naturally than box blur. Nearby pixels matter more than far away pixels.

**Why the assignment gives you cv2.getGaussianKernel:** Calculating the exact Gaussian values by hand is tedious math. The assignment lets you use this function JUST to generate the numbers. You still apply it yourself.

---

### c. Motion Blur (15x15)

The kernel has 1s only on the diagonal:

```
| 1  0  0  0  ... |
| 0  1  0  0  ... |
| 0  0  1  0  ... |
| 0  0  0  1  ... |
| ...          ... |
```

Divided by 15 to normalize.

**What it does:** Averages pixels along a diagonal line, simulating a camera moving diagonally during exposure. Makes the image look like it was taken while the camera was shaking.

---

### d. Laplacian Sharpening (3x3)

```
| -1  -1  -1 |
| -1   9  -1 |
| -1  -1  -1 |
```

**What it does:** Notice the center value is 9 and everything around it is -1. This ENHANCES the center pixel and SUBTRACTS the neighbors. The effect is that edges (where pixels change rapidly) get amplified, making the image look sharper.

**Why 9?** The surrounding values sum to -8. Center is 9. Total: 9 + (-8) = 1. This preserves overall brightness while boosting edges.

---

### e. Canny Edge Detection

This is a multi-step process, not a single kernel:

#### Step 1: Gaussian Smooth (5x5)

Blur the image first to reduce noise. Noise looks like tiny edges, so we smooth them out before detecting real edges.

#### Step 2: Sobel Kernels (finding edges)

Two 3x3 kernels that detect edges in different directions:

```
Gx (vertical edges):     Gy (horizontal edges):
| -1  0  1 |              | -1  -2  -1 |
| -2  0  2 |              |  0   0   0 |
| -1  0  1 |              |  1   2   1 |
```

- **Gx** detects vertical edges (changes from left to right)
- **Gy** detects horizontal edges (changes from top to bottom)

Apply both kernels separately, then combine:

- **Magnitude:** `M = sqrt(Gx² + Gy²)` — how strong is the edge?
- **Direction:** `theta = arctan(Gy / Gx)` — which way does the edge go?

#### Step 3: Non-Maximum Suppression

Edges from Step 2 are thick/blurry. This step thins them to 1 pixel wide by keeping only the strongest pixel along the edge direction.

#### Step 4: Thresholding

Set two thresholds (high and low):

- **Strong edges:** magnitude > high threshold → definitely an edge
- **Weak edges:** magnitude between low and high → maybe an edge (keep only if connected to a strong edge)
- **Below low threshold:** not an edge → set to 0

---

## Summary

| Filter    | Kernel                | Effect           |
| --------- | --------------------- | ---------------- |
| Box Blur  | All equal values      | Uniform blur     |
| Gaussian  | Bell curve weights    | Natural blur     |
| Motion    | Diagonal 1s           | Directional blur |
| Laplacian | Center=9, surround=-1 | Sharpening       |
| Canny     | Multiple steps        | Edge detection   |

## The Key Idea

All of these (except Canny) follow the same pattern:

1. Build a kernel (small matrix of numbers)
2. Slide it over every pixel
3. Multiply & sum to get the new pixel value

That's convolution. The **kernel values** determine what the filter does — blur, sharpen, detect edges, etc.
