"""Exhaustive learned-descriptor matching without PCA or uint8 conversion.

COLMAP 3.13's native SIFT matcher accepts only 128-D uint8 descriptors. This
explicit bridge keeps RDD's 256-D float vectors, applies mutual nearest-neighbor
and angular ratio/distance tests, and passes correspondences to COLMAP geometry.
The missing historical bridge cannot be recovered from the available notes.
"""
import numpy as np


def normalized(descriptors):
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Descriptors must be a finite N×D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero descriptor cannot be matched")
    return values / norms


def nearest_two(a, b, block_size):
    indices = np.zeros(len(a), dtype=np.int64)
    angles = np.zeros((len(a), 2), dtype=np.float32)
    for start in range(0, len(a), block_size):
        similarity = a[start:start+block_size] @ b.T
        # Stable argsort makes tied/duplicate descriptors reproducible.
        order = np.argsort(-similarity, axis=1, kind="stable")[:, :2]
        best = np.take_along_axis(similarity, order, axis=1)
        n = len(order)
        indices[start:start+n] = order[:, 0]
        angles[start:start+n] = np.arccos(np.clip(best, -1, 1))
    return indices, angles


def match_descriptors(first, second, ratio=0.8, max_distance=0.7, block_size=256):
    if not 0 < ratio < 1 or max_distance <= 0 or block_size < 1:
        raise ValueError("Invalid matching thresholds")
    first, second = normalized(first), normalized(second)
    if first.shape[1] != second.shape[1]:
        raise ValueError("Descriptor dimensions differ")
    if len(first) < 2 or len(second) < 2:
        return np.empty((0, 2), dtype=np.uint32)
    forward, distances = nearest_two(first, second, block_size)
    reverse, _ = nearest_two(second, first, block_size)
    rows = np.arange(len(first))
    valid = ((reverse[forward] == rows) & (distances[:, 0] < ratio * distances[:, 1]) &
             (distances[:, 0] <= max_distance))
    return np.column_stack((rows[valid], forward[valid])).astype(np.uint32)
