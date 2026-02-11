import math
import time

import cv2
import numpy as np

from hand_detector import HandDetector, draw_hand
from object_transformer import CubeController

# Canvas size
CANVAS_W, CANVAS_H = 1280, 720
# Picture-in-Picture size for webcam video
PIP_W, PIP_H = 384, 216

# Unit cube centered at origin: 8 vertices at ±0.5
CUBE_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)

# 12 edges: back face, front face, connecting
CUBE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # back face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # front face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # connecting
]


def rotation_matrix_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    # Rotation matrix around the X-axis:
    # [ 1     0       0   ]
    # [ 0   cosθ   -sinθ ]
    # [ 0   sinθ    cosθ ]
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    # Rotation matrix around the Y-axis:
    # [  cosθ   0   sinθ ]
    # [   0     1    0   ]
    # [ -sinθ   0   cosθ ]
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project_vertices(vertices, angle_x, angle_y, scale, cx, cy):
    """Rotate, scale, and orthographic-project vertices to 2D screen coords."""
    rot = rotation_matrix_x(angle_x) @ rotation_matrix_y(angle_y)
    rotated = (rot @ vertices.T).T
    points_2d = []
    for v in rotated:
        px = int(cx + v[0] * scale)
        py = int(cy + v[1] * scale)
        points_2d.append((px, py))
    return points_2d


def draw_cube(canvas, points_2d):
    """Draw wireframe cube edges."""
    for i, j in CUBE_EDGES:
        cv2.line(canvas, points_2d[i], points_2d[j], (80, 200, 255), 2, cv2.LINE_AA)


def main():
    detector = HandDetector()
    controller = CubeController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    start_ms = int(time.time() * 1000)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Flip frame to make webcam video looks like a mirror
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Detect hands
        # MediaPipe requires strictly increasing timestamps so it can track hands across frames
        timestamp_ms = int(time.time() * 1000) - start_ms
        gesture, landmarks = detector.detect(frame, timestamp_ms)

        # --- Pinch logic ---
        if landmarks:
            controller.update(landmarks)
            draw_hand(frame, landmarks, w, h)
        else:
            controller.pinching = False

        # --- Build canvas ---
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Draw centered 3D wireframe cube
        cx, cy = CANVAS_W // 2, CANVAS_H // 2
        scale = controller.cube_size
        points_2d = project_vertices(
            CUBE_VERTICES, controller.angle_x, controller.angle_y, scale, cx, cy
        )
        draw_cube(canvas, points_2d)

        # Size label
        label = f"{int(scale)}px"
        cv2.putText(
            canvas,
            label,
            (cx - 30, cy + int(scale * 0.5) + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
        )

        # Pinch indicator
        if controller.pinching:
            cv2.putText(
                canvas,
                "PINCHING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 120),
                2,
            )

        # Gesture label
        if gesture:
            cv2.putText(
                canvas,
                f"GESTURE: {gesture}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 100),
                2,
            )

        # --- PiP webcam feed ---
        pip = cv2.resize(frame, (PIP_W, PIP_H))
        pip_x = CANVAS_W - PIP_W - 10
        pip_y = CANVAS_H - PIP_H - 10
        canvas[pip_y : pip_y + PIP_H, pip_x : pip_x + PIP_W] = pip

        # PiP border
        cv2.rectangle(
            canvas,
            (pip_x - 1, pip_y - 1),
            (pip_x + PIP_W, pip_y + PIP_H),
            (100, 100, 100),
            1,
        )

        cv2.imshow("Gesture Control", canvas)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
