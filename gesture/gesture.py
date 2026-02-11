import time

import cv2
import numpy as np

from hand_detector import HandDetector, draw_hand
from object_transformer import PinchResizer

# Canvas size
CANVAS_W, CANVAS_H = 1280, 720
# Picture-in-Picture size for webcam video
PIP_W, PIP_H = 320, 240


def main():
    detector = HandDetector()
    resizer = PinchResizer()

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
        hands = detector.detect(frame, timestamp_ms)

        # --- Pinch logic ---
        if hands:
            resizer.update(hands[0])
            draw_hand(frame, hands[0], w, h)
        else:
            resizer.pinching = False

        # --- Build canvas ---
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Draw centered square
        sz = int(resizer.square_size)
        cx, cy = CANVAS_W // 2, CANVAS_H // 2
        x1, y1 = cx - sz // 2, cy - sz // 2
        x2, y2 = x1 + sz, y1 + sz
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 200, 255), -1)

        # Size label
        label = f"{sz}px"
        cv2.putText(
            canvas,
            label,
            (cx - 30, y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
        )

        # Pinch indicator
        if resizer.pinching:
            cv2.putText(
                canvas,
                "PINCHING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 120),
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
