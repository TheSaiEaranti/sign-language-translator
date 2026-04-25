"""
tests/test_models.py

Unit tests for architectures, datasets, and inference engine.
Run: pytest tests/ -v
"""

import numpy as np
import torch
import pytest

from data.dataset import LettersDataset, WordsDataset
from data.landmarks import LandmarkExtractor, HandFrame, NUM_LANDMARKS, LANDMARK_DIM
from models.architectures import (
    StaticGestureNet,
    DynamicGestureNet,
    MotionDetector,
    PositionalEncoding,
)


# ── Architecture tests ────────────────────────────────────────────────────────

def test_static_gesture_net_shape():
    model = StaticGestureNet(num_classes=26)
    x = torch.randn(4, 2, 21, 3)
    out = model(x)
    assert out.shape == (4, 26)


def test_static_gesture_net_param_count():
    model = StaticGestureNet(num_classes=26)
    n = sum(p.numel() for p in model.parameters())
    # Should be small — under 250K
    assert n < 250_000


def test_dynamic_gesture_net_shape():
    model = DynamicGestureNet(num_classes=10, d_model=64, num_layers=2)
    x = torch.randn(2, 30, 2, 21, 3)
    out = model(x)
    assert out.shape == (2, 10)


def test_dynamic_gesture_net_with_mask():
    model = DynamicGestureNet(num_classes=10, d_model=64, num_layers=2)
    x = torch.randn(2, 30, 2, 21, 3)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, 25:] = True  # padding for sample 0
    out = model(x, mask=mask)
    assert out.shape == (2, 10)


def test_dynamic_gesture_net_variable_length():
    """Model should handle variable sequence lengths."""
    model = DynamicGestureNet(num_classes=10, d_model=64, num_layers=2)
    for T in [10, 20, 30, 45]:
        x = torch.randn(1, T, 2, 21, 3)
        out = model(x)
        assert out.shape == (1, 10)


def test_positional_encoding():
    pe = PositionalEncoding(d_model=64, max_len=60)
    x = torch.zeros(2, 30, 64)
    out = pe(x)
    # PE should add unique values to each position
    assert not torch.allclose(out[:, 0], out[:, 1])


# ── Motion detector tests ─────────────────────────────────────────────────────

def test_motion_detector_static():
    md = MotionDetector(threshold=0.05)
    landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    for _ in range(10):
        moving = md.update(landmarks)
    assert not moving


def test_motion_detector_moving():
    md = MotionDetector(threshold=0.05)
    for i in range(10):
        landmarks = np.ones((2, 21, 3), dtype=np.float32) * i * 0.1
        moving = md.update(landmarks)
    assert moving


def test_motion_detector_reset_on_none():
    md = MotionDetector()
    md.update(np.zeros((2, 21, 3), dtype=np.float32))
    md.update(None)
    assert len(md._buffer) == 0


# ── Dataset tests ─────────────────────────────────────────────────────────────

def test_letters_dataset_synthetic():
    ds = LettersDataset(data_root=None, augment=True, num_classes=26)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (2, 21, 3)
    assert 0 <= y < 26


def test_words_dataset_synthetic():
    ds = WordsDataset(data_root=None, augment=True)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (30, 2, 21, 3)
    assert 0 <= y < len(ds.vocab)


def test_words_dataset_resample():
    ds = WordsDataset(seq_len=30)
    short_seq = np.random.randn(15, 2, 21, 3).astype(np.float32)
    resampled = ds._resample(short_seq)
    assert resampled.shape == (30, 2, 21, 3)


# ── Forward + backward integration ────────────────────────────────────────────

def test_static_model_trains():
    model = StaticGestureNet(num_classes=26)
    x = torch.randn(4, 2, 21, 3)
    y = torch.randint(0, 26, (4,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    # Verify all params have gradients
    for p in model.parameters():
        assert p.grad is not None


def test_dynamic_model_trains():
    model = DynamicGestureNet(num_classes=10, d_model=64, num_layers=2)
    x = torch.randn(2, 30, 2, 21, 3)
    y = torch.randint(0, 10, (2,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
