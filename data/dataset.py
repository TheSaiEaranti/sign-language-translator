"""
data/dataset.py

PyTorch Datasets for static (letters) and dynamic (words) gesture data.
Includes landmark-space augmentations: rotation, scale, jitter, dropout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.landmarks import NUM_LANDMARKS, LANDMARK_DIM


# ── Augmentations (operate on (T, 2, 21, 3) or (2, 21, 3) tensors) ────────────

def rotate_3d(points: np.ndarray, angle_range: float = 0.2) -> np.ndarray:
    """Random rotation around the y-axis (vertical) — simulates slight head tilt."""
    angle = random.uniform(-angle_range, angle_range)
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    return points @ R.T


def scale(points: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    s = random.uniform(*scale_range)
    return points * s


def jitter(points: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return points + np.random.normal(0, sigma, points.shape).astype(np.float32)


def landmark_dropout(points: np.ndarray, p: float = 0.05) -> np.ndarray:
    """Randomly zero out landmarks — simulates occlusion or detection failures."""
    mask = np.random.random(points.shape[:-1]) > p
    return points * mask[..., None].astype(np.float32)


def temporal_dropout(sequence: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Randomly zero out frames — simulates dropped frames in real capture."""
    mask = np.random.random(sequence.shape[0]) > p
    return sequence * mask[:, None, None, None].astype(np.float32)


# ── Datasets ──────────────────────────────────────────────────────────────────

class LettersDataset(Dataset):
    """
    Static-frame ASL letters dataset.
    Expects: data_root/{label}/*.npy where each .npy is shape (2, 21, 3).
    
    Falls back to synthetic data if no directory exists (for smoke testing).
    """

    def __init__(self, data_root: str | None = None, augment: bool = True, num_classes: int = 26):
        self.augment = augment
        self.num_classes = num_classes
        self.samples: list[tuple[np.ndarray, int]] = []

        if data_root and Path(data_root).exists():
            self._load_real(data_root)
        else:
            self._load_synthetic()

    def _load_real(self, data_root: str):
        root = Path(data_root)
        for label_dir in sorted(root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = ord(label_dir.name.upper()) - ord("A")
            if 0 <= label < self.num_classes:
                for npy_file in label_dir.glob("*.npy"):
                    points = np.load(npy_file)
                    self.samples.append((points, label))

    def _load_synthetic(self):
        """Fallback synthetic data — random landmarks per class for testing."""
        rng = np.random.RandomState(42)
        for class_idx in range(self.num_classes):
            class_mean = rng.randn(2, NUM_LANDMARKS, LANDMARK_DIM).astype(np.float32)
            for _ in range(20):  # 20 samples per class
                noise = rng.randn(2, NUM_LANDMARKS, LANDMARK_DIM).astype(np.float32) * 0.1
                self.samples.append((class_mean + noise, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        points, label = self.samples[idx]
        points = points.copy()
        if self.augment:
            points = rotate_3d(points)
            points = scale(points)
            points = jitter(points)
            points = landmark_dropout(points)
        return torch.from_numpy(points), label


class WordsDataset(Dataset):
    """
    Dynamic-sequence ASL words dataset.
    Expects: data_root/manifest.json with entries:
      [{"label": "hello", "sequence_path": "seqs/0001.npy"}, ...]
    Each sequence is shape (T, 2, 21, 3).
    """

    def __init__(
        self,
        data_root: str | None = None,
        augment: bool = True,
        seq_len: int = 30,
        vocab: list[str] | None = None,
    ):
        self.augment = augment
        self.seq_len = seq_len
        self.samples: list[tuple[np.ndarray, int]] = []
        self.vocab = vocab or ["hello", "thank_you", "yes", "no", "please", "sorry", "help", "love", "name", "good"]
        self.label_to_idx = {w: i for i, w in enumerate(self.vocab)}

        if data_root and (Path(data_root) / "manifest.json").exists():
            self._load_real(data_root)
        else:
            self._load_synthetic()

    def _load_real(self, data_root: str):
        root = Path(data_root)
        with open(root / "manifest.json") as f:
            manifest = json.load(f)
        for entry in manifest:
            label = self.label_to_idx.get(entry["label"])
            if label is None:
                continue
            seq = np.load(root / entry["sequence_path"])
            self.samples.append((seq, label))

    def _load_synthetic(self):
        rng = np.random.RandomState(7)
        for class_idx, _ in enumerate(self.vocab):
            base_motion = rng.randn(self.seq_len, 2, NUM_LANDMARKS, LANDMARK_DIM).astype(np.float32) * 0.5
            for _ in range(15):
                noise = rng.randn(*base_motion.shape).astype(np.float32) * 0.1
                self.samples.append((base_motion + noise, class_idx))

    def _resample(self, sequence: np.ndarray) -> np.ndarray:
        """Resample temporal sequence to fixed seq_len."""
        T = sequence.shape[0]
        if T == self.seq_len:
            return sequence
        # Linear interpolation in time
        idx = np.linspace(0, T - 1, self.seq_len)
        idx_floor = np.floor(idx).astype(int)
        idx_ceil = np.ceil(idx).astype(int).clip(max=T - 1)
        frac = (idx - idx_floor)[:, None, None, None]
        return (1 - frac) * sequence[idx_floor] + frac * sequence[idx_ceil]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        seq = self._resample(seq.astype(np.float32))
        if self.augment:
            seq = rotate_3d(seq)
            seq = scale(seq)
            seq = jitter(seq)
            seq = temporal_dropout(seq)
        return torch.from_numpy(seq), label
