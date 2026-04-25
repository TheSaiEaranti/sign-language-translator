"""data/landmarks.py — MediaPipe hand landmark extraction."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Import the legacy solutions API directly
try:
    from mediapipe.python.solutions import hands as mp_hands_module
except ImportError:
    import mediapipe as mp
    mp_hands_module = mp.solutions.hands

NUM_LANDMARKS = 21
LANDMARK_DIM = 3

@dataclass
class HandFrame:
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    timestamp: float = 0.0

    def to_tensor(self) -> np.ndarray:
        out = np.zeros((2, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        if self.left is not None: out[0] = self.left
        if self.right is not None: out[1] = self.right
        return out

    def has_hands(self) -> bool:
        return self.left is not None or self.right is not None


class LandmarkExtractor:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, static_image_mode=False):
        self._hands = mp_hands_module.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_rgb: np.ndarray) -> HandFrame:
        results = self._hands.process(frame_rgb)
        left, right = None, None
        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
                points = self._normalize(points)
                label = handedness.classification[0].label
                if label == "Left": left = points
                else: right = points
        return HandFrame(left=left, right=right)

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        wrist = points[0]
        translated = points - wrist
        ref_dist = np.linalg.norm(translated[9])
        if ref_dist < 1e-6: return translated
        return translated / ref_dist

    def close(self):
        self._hands.close()
