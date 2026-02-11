from hand_detector import is_pinching, hand_apparent_size, hand_centroid

SENSITIVITY = 3000
ROTATION_SENSITIVITY_X = 5.0
ROTATION_SENSITIVITY_Y = 5.0
MIN_CUBE, MAX_CUBE = 30, 600


class CubeController:
    def __init__(self, initial_size=150.0):
        self.cube_size = initial_size
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.pinching = False

        self._initial_hand_size = 0.0
        self._size_at_pinch_start = initial_size
        self._initial_centroid = (0.0, 0.0)
        self._angles_at_pinch_start = (0.0, 0.0)

    def update(self, landmarks):
        if is_pinching(landmarks):
            if not self.pinching:
                # Pinch just started — save baselines
                self.pinching = True
                self._initial_hand_size = hand_apparent_size(landmarks)
                self._size_at_pinch_start = self.cube_size
                self._initial_centroid = hand_centroid(landmarks)
                self._angles_at_pinch_start = (self.angle_x, self.angle_y)
            else:
                # Resize: depth proxy
                delta = self._initial_hand_size - hand_apparent_size(landmarks)
                self.cube_size = self._size_at_pinch_start + delta * SENSITIVITY
                self.cube_size = max(MIN_CUBE, min(MAX_CUBE, self.cube_size))

                # Rotate: hand movement
                cx, cy = hand_centroid(landmarks)
                ix, iy = self._initial_centroid
                dx = cx - ix
                dy = cy - iy
                ax0, ay0 = self._angles_at_pinch_start
                self.angle_y = ay0 - dx * ROTATION_SENSITIVITY_Y
                self.angle_x = ax0 + dy * ROTATION_SENSITIVITY_X
        else:
            self.pinching = False
