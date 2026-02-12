import numpy as np
import cv2

# 1. Load cuboid image
img = cv2.imread("cuboid.jpg")

# 2. Define 6 pairs of 2D-3D correspondences
# Cuboid: Focusrite Scarlett 2i2 — 180mm (W) x 47.5mm (H) x 117mm (D)
# Origin at front-bottom-left, X=right, Y=up, Z=back
correspondences = np.array(
    [
        [363, 1059, 0, 47.5, 0],  # Front-Top-Left
        [1199, 945, 180, 47.5, 0],  # Front-Top-Right
        [1181, 1125, 180, 0, 0],  # Front-Bottom-Right
        [383, 1280, 0, 0, 0],  # Front-Bottom-Left
        [277, 1051, 0, 0, 117],  # Back-Bottom-Left
        [275, 911, 0, 47.5, 117],  # Back-Top-Left
        [938, 833, 180, 47.5, 117],  # Back-Top-Right
    ]
)

pts_2d = correspondences[:, :2]
pts_3d = correspondences[:, 2:]
N = pts_2d.shape[0]

# 3. Compute camera matrix P using DLT
A = np.zeros((2 * N, 12))
for i in range(N):
    X, Y, Z = pts_3d[i]
    u, v = pts_2d[i]
    A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
    A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

_, _, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)

print("Camera matrix P:")
print(P)

# 4. (Optional) Draw the cuboid edges using P
# All 8 corners of the cuboid
corners_3d = np.array(
    [
        [0, 0, 0],
        [180, 0, 0],
        [180, 47.5, 0],
        [0, 47.5, 0],
        [0, 0, 117],
        [180, 0, 117],
        [180, 47.5, 117],
        [0, 47.5, 117],
    ]
)

# 12 edges as pairs of corner indices
edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # front face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # back face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # connecting edges
]

# Project corners
corners_h = np.hstack([corners_3d, np.ones((8, 1))])
projected = (P @ corners_h.T).T
projected = projected[:, :2] / projected[:, 2:3]

# Draw edges
for i, j in edges:
    p1 = (int(round(projected[i][0])), int(round(projected[i][1])))
    p2 = (int(round(projected[j][0])), int(round(projected[j][1])))
    cv2.line(img, p1, p2, (0, 255, 0), 3)

cv2.imwrite("cuboid_part2.jpeg", img)
print("Saved cuboid_part2.jpeg")

# 5. Fun: Draw 3D text "Han Nguyen" floating above the cuboid
def project(pt3d):
    h = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
    p = P @ h
    return (int(round(p[0] / p[2])), int(round(p[1] / p[2])))

def make_letter(lines, ox, oy, oz, s=1.0):
    """Offset and scale a letter's line segments."""
    return [([a[0]*s+ox, a[1]*s+oy, a[2]*s+oz], [b[0]*s+ox, b[1]*s+oy, b[2]*s+oz]) for a, b in lines]

# Base letter shapes (defined in a 0-4 wide, 0-6 tall grid)
L_H = [([0,0,0],[0,6,0]), ([4,0,0],[4,6,0]), ([0,3,0],[4,3,0])]
L_a = [([0,0,0],[1.5,4,0]), ([1.5,4,0],[3,0,0]), ([0.75,2,0],[2.25,2,0])]
L_n = [([0,0,0],[0,4,0]), ([0,4,0],[3,4,0]), ([3,4,0],[3,0,0])]

L_N = [([0,0,0],[0,6,0]), ([0,6,0],[4,0,0]), ([4,0,0],[4,6,0])]
L_g = [([0,0,0],[3,0,0]), ([3,0,0],[3,4,0]), ([0,0,0],[0,4,0]), ([0,4,0],[3,4,0]), ([3,0,0],[3,-2,0]), ([3,-2,0],[0,-2,0])]
L_u = [([0,4,0],[0,0,0]), ([0,0,0],[3,0,0]), ([3,0,0],[3,4,0])]
L_y = [([0,4,0],[0,2,0]), ([0,2,0],[3,2,0]), ([3,4,0],[3,0,0]), ([3,0,0],[0,0,0])]
L_e = [([0,0,0],[3,0,0]), ([0,0,0],[0,4,0]), ([0,4,0],[3,4,0]), ([0,2,0],[3,2,0])]

# Layout: "Han" on top row, "Nguyen" on bottom row
# Positioned above the cuboid top face
s = 5  # scale factor
z = -60  # Z position (in front of the cuboid, front face is Z=0)

# Top row: "Han"
top_y = 90
letters_top = [
    make_letter(L_H, 20, top_y, z, s),
    make_letter(L_a, 50, top_y, z, s),
    make_letter(L_n, 75, top_y, z, s),
]

# Bottom row: "Nguyen"
bot_y = 40
letters_bot = [
    make_letter(L_N, 5, bot_y, z, s),
    make_letter(L_g, 35, bot_y, z, s),
    make_letter(L_u, 58, bot_y, z, s),
    make_letter(L_y, 81, bot_y, z, s),
    make_letter(L_e, 104, bot_y, z, s),
    make_letter(L_n, 127, bot_y, z, s),
]

img2 = cv2.imread("cuboid.jpg")
# Redraw cuboid edges
for i, j in edges:
    p1 = (int(round(projected[i][0])), int(round(projected[i][1])))
    p2 = (int(round(projected[j][0])), int(round(projected[j][1])))
    cv2.line(img2, p1, p2, (0, 255, 0), 3)

# Draw numbered points at each corner
for i, pt in enumerate(projected):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.circle(img2, (x, y), 12, (0, 255, 255), -1)
    cv2.putText(img2, str(i + 1), (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

# Draw all letters
for letter in letters_top + letters_bot:
    for start, end in letter:
        cv2.line(img2, project(start), project(end), (255, 100, 0), 3)

cv2.imwrite("cuboid_fun.jpeg", img2)
print("Saved cuboid_fun.jpeg")
