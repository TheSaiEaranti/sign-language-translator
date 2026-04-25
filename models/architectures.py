"""
models/architectures.py

Two custom architectures:

1. StaticGestureNet — small MLP for single-frame letter classification.
   Input:  (B, 2, 21, 3) — two hands, 21 landmarks, xyz
   Output: (B, 26)        — letter logits

2. DynamicGestureNet — Transformer encoder over temporal landmark sequences.
   Input:  (B, T, 2, 21, 3) — sequence of frames
   Output: (B, num_words)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── StaticGestureNet ──────────────────────────────────────────────────────────

class StaticGestureNet(nn.Module):
    """
    MLP for static letter classification. Tiny (~50K params), <2ms inference.
    Uses skip connections and layer norm for training stability.
    """

    def __init__(self, num_classes: int = 26, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        in_dim = 2 * 21 * 3  # 126

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 21, 3) → (B, 126)
        B = x.shape[0]
        x = x.reshape(B, -1)

        h = self.input_proj(x)
        h = self.norm1(h)

        h = h + self.block1(h)
        h = self.norm2(h)

        h = h + self.block2(h)
        h = self.norm3(h)

        return self.classifier(h)


# ── DynamicGestureNet ─────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the temporal axis."""

    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FrameEncoder(nn.Module):
    """Per-frame landmark encoder: (2, 21, 3) → (d_model,)."""

    def __init__(self, d_model: int):
        super().__init__()
        in_dim = 2 * 21 * 3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 2, 21, 3)
        B, T = frames.shape[:2]
        flat = frames.reshape(B, T, -1)
        return self.proj(flat)  # (B, T, d_model)


class DynamicGestureNet(nn.Module):
    """
    Transformer encoder for dynamic gesture (word) classification.
    Architecture:
      - FrameEncoder: per-frame landmarks → embeddings
      - Positional encoding
      - N transformer encoder layers with multi-head attention
      - Mean pooling over time + classifier head

    ~500K params at default config, runs in ~10ms on CPU.
    """

    def __init__(
        self,
        num_classes: int = 100,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # sequence: (B, T, 2, 21, 3)
        x = self.frame_encoder(sequence)         # (B, T, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # Mean pool over time, ignoring padded positions
        if mask is not None:
            keep = (~mask).float().unsqueeze(-1)
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)


# ── Motion detector (heuristic, no training) ─────────────────────────────────

class MotionDetector:
    """
    Lightweight motion detector to route frames between letter and word models.
    Tracks landmark velocity over a short window — high motion → words, low → letters.
    """

    def __init__(self, threshold: float = 0.05, window: int = 5):
        self.threshold = threshold
        self.window = window
        self._buffer: list = []

    def update(self, landmarks) -> bool:
        """Returns True if motion detected (use word model)."""
        if landmarks is None:
            self._buffer.clear()
            return False

        self._buffer.append(landmarks.flatten())
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        if len(self._buffer) < 2:
            return False

        velocities = []
        for i in range(1, len(self._buffer)):
            v = ((self._buffer[i] - self._buffer[i - 1]) ** 2).sum() ** 0.5
            velocities.append(v)

        avg_velocity = sum(velocities) / len(velocities)
        return avg_velocity > self.threshold

    def reset(self):
        self._buffer.clear()
