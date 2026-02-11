from hand_detector import is_pinching, hand_apparent_size

SENSITIVITY = 3000  # pixels of square size change per unit of normalized hand-size delta
MIN_SQUARE, MAX_SQUARE = 30, 600


class PinchResizer:
    def __init__(self, initial_size=150.0):
        self.square_size = initial_size
        self.pinching = False
        self._initial_hand_size = 0.0
        self._size_at_pinch_start = initial_size

    def update(self, landmarks):
        if is_pinching(landmarks):
            cur_hand_size = hand_apparent_size(landmarks)
            if not self.pinching:
                # Pinch just started
                self.pinching = True
                self._initial_hand_size = cur_hand_size
                self._size_at_pinch_start = self.square_size
            else:
                # Ongoing pinch — use apparent hand size as depth proxy
                delta = self._initial_hand_size - cur_hand_size
                self.square_size = self._size_at_pinch_start + delta * SENSITIVITY
                self.square_size = max(MIN_SQUARE, min(MAX_SQUARE, self.square_size))
        else:
            self.pinching = False
