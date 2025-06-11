#!/usr/bin/env python
"""
grid_former.py – Unified GRID-Former + ARC utilities

Combines:
1. ARCGridValidator  – lightweight palette validation / correction
2. GRID-Former       – transformer architecture for ARC tasks
"""

import os
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("GRID-Former")

# -----------------------------------------------------------------------------#
# ARC grid-validation helpers
# -----------------------------------------------------------------------------#
class ARCGridValidator:
    """Validate / correct color indices (0-9) and keep correction stats."""

    def __init__(self, color_correction_threshold: float = 0.8, max_history: int = 100):
        self.color_correction_threshold = color_correction_threshold
        self.arc_palette = list(range(10))
        self.correction_history: List[Tuple[Any, Any, float]] = []
        self.max_history = max_history

    # --- single value --------------------------------------------------------#
    def _record(self, original: Any, corrected: Any) -> None:
        if len(self.correction_history) >= self.max_history:
            self.correction_history.pop(0)
        self.correction_history.append((original, corrected, 1.0))

    def validate_color(self, value: Any) -> int:
        try:
            v = int(value)
            if v not in self.arc_palette:
                logger.warning(f"Color {v} out of bounds; clamped")
                c = max(0, min(9, v))
                self._record(v, c)
                return c
            return v
        except (ValueError, TypeError):
            logger.warning(f"Invalid color {value}; default -> 0")
            self._record(value, 0)
            return 0

    # --- numpy grid ----------------------------------------------------------#
    def correct_grid_colors(self, grid: np.ndarray) -> Tuple[np.ndarray, float]:
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        corrected = grid.copy()
        corrections = 0

        for idx in np.ndindex(grid.shape):
            o = grid[idx]
            v = self.validate_color(o)
            if o != v:
                corrected[idx] = v
                corrections += 1

        conf = 1.0 - corrections / grid.size if grid.size else 1.0
        return corrected, conf

    # --- pytorch tensor (in-place safe) --------------------------------------#
    def correct_tensor_colors(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clamp(0, 9)  # fast path
        return tensor

    # --- stats ----------------------------------------------------------------#
    def get_correction_patterns(self) -> Dict[int, int]:
        if not self.correction_history:
            return {}
        counts: Dict[Tuple[Any, Any], int] = {}
        for o, c, _ in self.correction_history:
            counts[(o, c)] = counts.get((o, c), 0) + 1
        patterns: Dict[int, int] = {}
        for o, _ in self.correction_history:
            best = max(
                ((c, n) for (oo, c), n in counts.items() if oo == o),
                key=lambda x: x[1],
                default=None,
            )
            if best:
                patterns[o] = best[0]
        return patterns


# -----------------------------------------------------------------------------#
# GRID-Former constants
# -----------------------------------------------------------------------------#
MAX_GRID_SIZE = 30
NUM_COLORS = 10
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

# -----------------------------------------------------------------------------#
# Model building blocks
# -----------------------------------------------------------------------------#
class GridPositionEncoding(nn.Module):
    """2-D sinusoidal positional encoding"""

    def __init__(self, hidden_dim: int, max_grid_size: int = MAX_GRID_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_grid_size, max_grid_size, hidden_dim)
        for i in range(max_grid_size):
            for j in range(max_grid_size):
                for k in range(0, hidden_dim, 2):
                    theta = (i * 10000 ** (-k / hidden_dim)) + (j * 10000 ** (-(k + 1) / hidden_dim))
                    pe[i, j, k] = math.sin(theta)
                    if k + 1 < hidden_dim:
                        pe[i, j, k + 1] = math.cos(theta)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        b, *_ = x.shape
        h, w = shape
        pos = self.pe[:h, :w, :].reshape(1, h * w, -1).repeat(b, 1, 1)
        return self.dropout(x + pos)


class GridEmbedding(nn.Module):
    """Color index -> embedding, with optional validation"""

    def __init__(self, hidden_dim: int, num_colors: int = NUM_COLORS, validator: Optional[ARCGridValidator] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_colors = num_colors
        self.validator = validator

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if self.validator is not None and not grid.is_cuda:
            # small CPU overhead; skip if tensor is on GPU
            grid_np = grid.cpu().numpy()
            grid_np = self.validator.correct_tensor_colors(torch.tensor(grid_np)).numpy()
            grid = torch.from_numpy(grid_np).to(grid.device)

        flat = grid.reshape(grid.size(0), -1).clamp(0, self.num_colors - 1)
        emb = self.norm(self.embedding(flat))
        return emb


class GridPatternRecognitionLayer(nn.Module):
    """Simple symmetry / transform detection"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.sym = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2))
        self.trf = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2))
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, *_):
        out = self.mix(torch.cat([self.sym(x), self.trf(x)], dim=2))
        return self.norm(x + out)


class GridMultiHeadAttention(nn.Module):
    """Self-attention"""

    def __init__(self, hidden_dim: int, heads: int = NUM_HEADS, dropout: float = DROPOUT):
        super().__init__()
        assert hidden_dim % heads == 0
        self.h = heads
        self.d = hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, n, self.h, self.d).transpose(1, 2)
        k = k.reshape(b, n, self.h, self.d).transpose(1, 2)
        v = v.reshape(b, n, self.h, self.d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        att = F.softmax(att, -1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).reshape(b, n, c)
        out = self.drop(self.o(out))
        return self.norm(out + x)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(self.net(x) + x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = NUM_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.attn = GridMultiHeadAttention(hidden_dim, heads, dropout)
        self.ffn = FeedForward(hidden_dim, dropout)

    def forward(self, x):
        return self.ffn(self.attn(x))


class GridTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int = NUM_LAYERS, heads: int = NUM_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(hidden_dim, heads, dropout) for _ in range(layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class GridTransformationDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_colors: int = NUM_COLORS):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )

    def forward(self, x, out_shape: Tuple[int, int]):
        b, seq, _ = x.shape
        logits = self.decode(x)  # (b, seq, C)
        h, w = out_shape
        C = logits.size(-1)

        # reshape to temp grid then interpolate
        temp_size = int(seq ** 0.5) or 1
        temp_h = temp_size
        temp_w = (seq + temp_h - 1) // temp_h
        pad = temp_h * temp_w - seq
        if pad:
            logits = torch.cat([logits, logits.new_zeros(b, pad, C)], 1)
        grid = logits.reshape(b, temp_h, temp_w, C).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
        return grid.permute(0, 2, 3, 1)


# -----------------------------------------------------------------------------#
# Full model
# -----------------------------------------------------------------------------#
class GRID_Former(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        layers: int = NUM_LAYERS,
        heads: int = NUM_HEADS,
        num_colors: int = NUM_COLORS,
        max_grid_size: int = MAX_GRID_SIZE,
        dropout: float = DROPOUT,
        validator: Optional[ARCGridValidator] = None,
    ):
        super().__init__()
        self.embed = GridEmbedding(hidden_dim, num_colors, validator)
        self.pos = GridPositionEncoding(hidden_dim, max_grid_size, dropout)
        self.pattern = GridPatternRecognitionLayer(hidden_dim)
        self.enc = GridTransformerEncoder(hidden_dim, layers, heads, dropout)
        self.dec = GridTransformationDecoder(hidden_dim, num_colors)

    def forward(self, grid: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None):
        if target_shape is None:
            target_shape = grid.shape[1:3]
        b, h, w = grid.shape
        x = self.embed(grid)
        x = self.pos(x, (h, w))
        x = self.pattern(x, (h, w))
        x = self.enc(x)
        return self.dec(x, target_shape)

    # convenience helpers -----------------------------------------------------#
    def predict(self, grid: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None):
        self.eval()
        with torch.no_grad():
            logits = self(grid, target_shape)
            return logits.argmax(-1)

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state": self.state_dict()}, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_from_file(cls, path: str, device: str = "cpu") -> "GRID_Former":
        ckpt = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(ckpt["state"])
        return model.to(device)


# -----------------------------------------------------------------------------#
# Tiny smoke-test
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    validator = ARCGridValidator()
    model = GRID_Former(validator=validator)
    x = torch.randint(0, 10, (2, 6, 7))
    out = model(x, (8, 8))
    print("output:", out.shape)
