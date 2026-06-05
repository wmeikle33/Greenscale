from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d
from skimage.morphology import skeletonize


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    """Convert mask to 0/1 uint8."""
    if mask.max() > 1:
        mask = mask > 127
    else:
        mask = mask > 0
    return mask.astype(np.uint8)


def distance_layer(mask: np.ndarray) -> np.ndarray:
    """Root thickness / distance from background."""
    return distance_transform_edt(mask > 0).astype(np.float32)


def skeleton_layer(mask: np.ndarray) -> np.ndarray:
    """Thin root topology skeleton."""
    return skeletonize(mask > 0).astype(np.float32)


def depth_density_layer(mask: np.ndarray) -> np.ndarray:
    """Root density by vertical depth."""
    density = mask.mean(axis=1, keepdims=True)
    return np.repeat(density, mask.shape[1], axis=1).astype(np.float32)


def width_density_layer(mask: np.ndarray) -> np.ndarray:
    """Root density across horizontal width."""
    density = mask.mean(axis=0, keepdims=True)
    return np.repeat(density, mask.shape[0], axis=0).astype(np.float32)


def branch_layer(mask: np.ndarray) -> np.ndarray:
    """Approximate branch/crossing hotspots from skeleton neighbors."""
    skel = skeletonize(mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = convolve2d(skel, kernel, mode="same", boundary="fill", fillvalue=0)

    return ((skel == 1) & (neighbors >= 4)).astype(np.float32)


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize one feature channel to 0–1."""
    channel = channel.astype(np.float32)
    max_val = channel.max()

    if max_val == 0:
        return channel

    return channel / max_val


def build_25d(mask: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Build a 2.5D feature stack from a binary root mask.

    Output shape:
        H x W x 6

    Channels:
        0 binary mask
        1 distance transform
        2 skeleton
        3 depth density
        4 width density
        5 branch hotspots
    """
    mask = ensure_binary(mask)

    channels = [
        mask.astype(np.float32),
        distance_layer(mask),
        skeleton_layer(mask),
        depth_density_layer(mask),
        width_density_layer(mask),
        branch_layer(mask),
    ]

    if normalize:
        channels = [normalize_channel(c) for c in channels]

    return np.stack(channels, axis=-1).astype(np.float32)


def save_25d(mask: np.ndarray, output_path: str | Path) -> Path:
    """Create and save a 2.5D `.npy` file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = build_25d(mask)
    np.save(output_path, features)

    return output_path
