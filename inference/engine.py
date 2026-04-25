"""
inference/engine.py

Real-time inference engine. Routes frames between letter and word models
based on motion detection. Maintains a sliding buffer for dynamic gestures.

Latency budget (target):
  - Hand tracking:       ~15ms (MediaPipe)
  - Letter inference:    ~2ms
  - Word inference:      ~10ms
  - Smoothing/dispatch:  ~3ms
  Total:                 ~30ms (33fps achievable)
"""

from __future__ import annotations

import time
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data.landmarks import LandmarkExtractor, HandFrame
from models.architectures import StaticGestureNet, DynamicGestureNet, MotionDetector


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_VOCAB = ["hello", "thank_you", "yes", "no", "please", "sorry", "help", "love", "name", "good"]


@dataclass
class Prediction:
    label: str
    confidence: float
    mode: str  # "letter" | "word" | "none"
    latency_ms: float


class InferenceEngine:
    """
    Hybrid inference engine for real-time sign language translation.
    
    Usage:
        engine = InferenceEngine(letters_ckpt="...", words_ckpt="...")
        
        for frame in webcam:
            pred = engine.predict(frame)
            if pred:
                print(pred.label, pred.confidence)
    """

    def __init__(
        self,
        letters_ckpt: str | None = None,
        words_ckpt: str | None = None,
        seq_len: int = 30,
        device: str = "cpu",
        vocab: list[str] | None = None,
        motion_threshold: float = 0.05,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.0,
    ):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.vocab = vocab or DEFAULT_VOCAB
        self.confidence_threshold = confidence_threshold

        self.extractor = LandmarkExtractor(min_detection_confidence=0.6)
        self.motion = MotionDetector(threshold=motion_threshold)

        # Letters model
        self.letters_model = StaticGestureNet(num_classes=26).to(self.device)
        if letters_ckpt and Path(letters_ckpt).exists():
            ckpt = torch.load(letters_ckpt, map_location=self.device)
            self.letters_model.load_state_dict(ckpt["model_state_dict"])
        self.letters_model.eval()

        # Words model
        self.words_model = DynamicGestureNet(num_classes=len(self.vocab)).to(self.device)
        if words_ckpt and Path(words_ckpt).exists():
            ckpt = torch.load(words_ckpt, map_location=self.device)
            self.words_model.load_state_dict(ckpt["model_state_dict"])
        self.words_model.eval()

        # Sliding buffer for dynamic gestures
        self._sequence_buffer: deque = deque(maxlen=seq_len)
        # Smoothing buffer for stable predictions
        self._prediction_buffer: deque = deque(maxlen=smoothing_window)

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray) -> Optional[Prediction]:
        """
        Process a single frame. Returns a Prediction or None if no hands detected.
        """
        start = time.monotonic()

        hand_frame = self.extractor.extract(frame_rgb)
        if not hand_frame.has_hands():
            self._sequence_buffer.clear()
            self._prediction_buffer.clear()
            self.motion.reset()
            return None

        landmarks = hand_frame.to_tensor()  # (2, 21, 3)
        self._sequence_buffer.append(landmarks)

        moving = self.motion.update(landmarks)

        if moving and len(self._sequence_buffer) >= self.seq_len:
            pred = self._predict_word()
            mode = "word"
        else:
            pred = self._predict_letter(landmarks)
            mode = "letter"

        # Smooth predictions over recent frames
        self._prediction_buffer.append((pred["label"], pred["confidence"]))
        smoothed = self._smooth_predictions()

        latency_ms = (time.monotonic() - start) * 1000
        return Prediction(
            label=smoothed["label"],
            confidence=smoothed["confidence"],
            mode=mode,
            latency_ms=latency_ms,
        )

    def _predict_letter(self, landmarks: np.ndarray) -> dict:
        x = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)
        logits = self.letters_model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return {"label": LETTERS[idx.item()], "confidence": conf.item()}

    def _predict_word(self) -> dict:
        sequence = np.stack(list(self._sequence_buffer), axis=0)  # (T, 2, 21, 3)
        x = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        logits = self.words_model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return {"label": self.vocab[idx.item()], "confidence": conf.item()}

    def _smooth_predictions(self) -> dict:
        """Majority vote over recent predictions, weighted by confidence."""
        labels = [p[0] for p in self._prediction_buffer]
        most_common, _ = Counter(labels).most_common(1)[0]
        confs = [p[1] for p in self._prediction_buffer if p[0] == most_common]
        avg_conf = sum(confs) / len(confs)
        return {"label": most_common, "confidence": avg_conf}

    def reset(self):
        self._sequence_buffer.clear()
        self._prediction_buffer.clear()
        self.motion.reset()
