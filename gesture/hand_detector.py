import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def get_model_path():
    cache_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / "gesture_recognizer.task"
    if not model_path.exists():
        print("Downloading gesture_recognizer model...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    return str(model_path)


def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, pts[i], pts[j], (180, 140, 255), 2, cv2.LINE_AA)

    for px, py in pts:
        cv2.circle(frame, (px, py), 5, (100, 255, 200), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


class HandDetector:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = get_model_path()
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def detect(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        if result.hand_landmarks:
            gesture = result.gestures[0][0].category_name if result.gestures else "None"
            return gesture, result.hand_landmarks[0]
        return None, None

    def close(self):
        self._recognizer.close()


PINCH_THRESHOLD = 0.1


def pinch_distance(landmarks):
    """Return distance between thumb tip (4) and index tip (8) in normalized coords."""
    thumb = landmarks[4]
    index = landmarks[8]
    return ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5


def hand_apparent_size(landmarks):
    """Distance from wrist (0) to middle finger MCP (9) — proxy for how far the hand is.
    Hand close to camera → large value. Hand far from camera → small value."""
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    return ((wrist.x - mid_mcp.x) ** 2 + (wrist.y - mid_mcp.y) ** 2) ** 0.5


def is_pinching(landmarks):
    """Return whether the hand is in a pinch gesture."""
    return pinch_distance(landmarks) < PINCH_THRESHOLD


def hand_centroid(landmarks):
    """Midpoint of thumb tip (4) and index tip (8) in normalized coords."""
    thumb = landmarks[4]
    index = landmarks[8]
    return ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2)
