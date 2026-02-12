import numpy as np
import cv2

# 1. Build the camera matrix P
# Load 2D-3D correspondences
data = np.loadtxt("bunny.txt")
pts_2d = data[:, :2]  # (N, 2) — u, v
pts_3d = data[:, 2:]  # (N, 3) — X, Y, Z

N = pts_2d.shape[0]

# Build the DLT matrix A (2N x 12)
A = np.zeros((2 * N, 12))
for i in range(N):
    X, Y, Z = pts_3d[i]
    u, v = pts_2d[i]
    A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
    A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

# Solve Ap = 0 via SVD
# P is the last row of V^T
_, _, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)

print("Camera matrix P:")
print(P)

# 2. Project surface points onto the image
img = cv2.imread("bunny.jpeg")
surface_pts = np.load("bunny_pts.npy")  # (M, 3)

# Convert to homogeneous and project
surface_h = np.hstack([surface_pts, np.ones((surface_pts.shape[0], 1))])  # (M, 4)
projected = (P @ surface_h.T).T  # (M, 3)
projected = projected[:, :2] / projected[:, 2:3]  # dehomogenize to (M, 2)

# Draw orange dots on the image
for pt in projected:
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(img, (x, y), 3, (0, 80, 255), -1)  # orange

cv2.imwrite("bunny_part1-2.jpeg", img)
print("\nSaved bunny_part1-2.jpeg")

# 3. Project bounding box edges onto the image
img2 = cv2.imread("bunny.jpeg")
edges = np.load("bunny_bd.npy")  # (12, 6) — each row is two 3D points

for edge in edges:
    p1_3d = np.array([edge[0], edge[1], edge[2], 1.0])
    p2_3d = np.array([edge[3], edge[4], edge[5], 1.0])

    p1_proj = P @ p1_3d
    p2_proj = P @ p2_3d

    p1_2d = (int(round(p1_proj[0] / p1_proj[2])), int(round(p1_proj[1] / p1_proj[2])))
    p2_2d = (int(round(p2_proj[0] / p2_proj[2])), int(round(p2_proj[1] / p2_proj[2])))

    cv2.line(img2, p1_2d, p2_2d, (0, 255, 80), 4)

cv2.imwrite("bunny_part1-3.jpeg", img2)
print("Saved bunny_part1-3.jpeg")
