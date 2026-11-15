"""Patch extraction shared by dataset loaders and preparation scripts."""

from __future__ import annotations

import numpy as np


def make_patches(features: np.ndarray, labels: np.ndarray, window_size: int):
    """Extract zero-padded square patches for every non-background label."""
    if features.ndim != 3:
        raise ValueError("features must have shape (height, width, channels)")
    if labels.shape != features.shape[:2]:
        raise ValueError("labels must match the spatial dimensions of features")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")

    margin = window_size // 2
    padded = np.pad(features, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    rows, columns = np.nonzero(labels)
    class_ids = labels[rows, columns].astype(np.int64)
    index = np.column_stack((class_ids, rows, columns))

    patches = np.empty(
        (len(index), window_size, window_size, features.shape[2]),
        dtype=np.float32,
    )
    for position, (_, row, column) in enumerate(index):
        patches[position] = padded[row : row + window_size, column : column + window_size]

    return patches, class_ids, index
