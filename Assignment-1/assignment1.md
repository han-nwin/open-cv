---
# CS 4391 – Spring 2026

## Assignment 1 – Single-View Camera Projection

**Due Date:**  
**Feb 22nd, 2026 — 11:59 PM**

---

## Overview

In this assignment, your goal is to compute the **camera matrix P** from **2D–3D point correspondences**.

---

## Part I (70 Points)

You are given:

- A picture of the **Stanford Bunny** (`bunny.jpeg`)
- A text file containing **2D–3D point correspondences** (`bunny.txt`)

### 2D–3D Correspondences Format

The text file contains multiple rows.

- Each row represents **one pair of 2D–3D correspondences**
- The **first 2 numbers** are the **2D coordinates** on the image
- The **next 3 numbers** are the corresponding **3D coordinates** in world space

---

### Input Image

- Input image of the Stanford Bunny
- Annotated 2D points on the image

---

### Instructions

1. **Compute the camera matrix `P`** using the provided 2D–3D correspondences.

2. We provide a set of **3D surface points** in:

```

bunny_pts.npy

```

- Project these points to the image using your calculated `P`
- See the example below
- Demo color is **red** (you may choose any color)
- Drawing straight lines between pairs of points is **optional**

3. We provide the **12 edges of the bounding box** in:

```

bunny_bd.npy

```

- Each line contains **6 numbers**
- Every **3 numbers denote one 3D point**
- Project these points to the image
- Draw the cuboid by drawing a straight line between each pair of points
- See the example below
- Demo color is **blue** (you may choose any color)

---

### Visual References

- Surface Points (projected onto image)
- Bounding Box (projected cuboid)

---

## Part II – Cuboid Experiment (30 Points)

### Instructions

1. Find (or capture) **one image of a cuboid**

2. Come up with a **3D coordinate system** by:

- Measuring relative dimensions of the cuboid

3. Annotate **6 pairs of 2D–3D point correspondences**

4. **Compute the camera matrix `P`** using your annotated correspondences

5. _(Optional)_

- Draw the edges of the cuboid using your calculated `P`
- Or do something fun!

---

## Submission Instructions

- **Submit screenshots only**
- Upload to **eLearning**

---
