"""Convert Houston HSI/LiDAR `.mat` files into patch-based NumPy datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from my_cil.data import make_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_houston"))
    parser.add_argument("--window-size", type=int, default=11)
    return parser.parse_args()


def normalize_channels(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    minimum = features.min(axis=(0, 1), keepdims=True)
    shifted = features - minimum
    maximum = shifted.max(axis=(0, 1), keepdims=True)
    return np.divide(shifted, maximum, out=np.zeros_like(shifted), where=maximum != 0)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    hsi = sio.loadmat(data_dir / "HSI.mat")["spectraldata"]
    lidar = sio.loadmat(data_dir / "LIDAR.mat")["LiDAR_data"]
    features = normalize_channels(np.concatenate([hsi, lidar[..., None]], axis=2))
    train_labels_map = sio.loadmat(data_dir / "train_labels.mat")["train_labels"]
    test_labels_map = sio.loadmat(data_dir / "test_labels.mat")["test_labels"]

    train_patches, train_labels, _ = make_patches(features, train_labels_map, args.window_size)
    test_patches, test_labels, _ = make_patches(features, test_labels_map, args.window_size)

    np.save(data_dir / "train_patches.npy", train_patches)
    np.save(data_dir / "train_labels.npy", train_labels)
    np.save(data_dir / "test_patches.npy", test_patches)
    np.save(data_dir / "test_labels.npy", test_labels)
    print(f"Saved {len(train_labels)} training and {len(test_labels)} test patches to {data_dir}")


if __name__ == "__main__":
    main()
