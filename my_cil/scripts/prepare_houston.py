"""Convert Houston HSI/LiDAR `.mat` files into patch-based NumPy datasets.

Defaults reproduce the original behaviour (all 144 bands, train/test split
taken from the label files). Paper setting (BiCM-Prompt, TIP 2026, Sec. IV):

    python -m scripts.prepare_houston --raw-dir data_houston/raw/Houston2013 \
        --hsi-file HSI.mat --lidar-file LiDAR.mat \
        --train-file TRLabel.mat --test-file TSLabel.mat \
        --pca 40 --split random --train-per-class 180 --seed 0 \
        --out-dir data_houston/paper_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from my_cil.data import make_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data_houston"),
                        help="default for --raw-dir and --out-dir")
    parser.add_argument("--raw-dir", type=Path, help="folder with the .mat files (default: --data-dir)")
    parser.add_argument("--out-dir", type=Path, help="where to write *.npy (default: --data-dir)")
    parser.add_argument("--hsi-file", default="HSI.mat")
    parser.add_argument("--lidar-file", default="LIDAR.mat")
    parser.add_argument("--train-file", default="train_labels.mat")
    parser.add_argument("--test-file", default="test_labels.mat")
    parser.add_argument("--window-size", type=int, default=11)
    parser.add_argument("--pca", type=int, default=0, help="number of HSI PCA components (0 = keep all bands)")
    parser.add_argument("--split", choices=("given", "random"), default="given",
                        help="'given': use train/test label files; 'random': pool both and draw "
                             "--train-per-class training pixels per class, the rest is test")
    parser.add_argument("--train-per-class", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0, help="seed of the random split")
    return parser.parse_args()


def load_mat_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    """Load the array from a .mat file, trying known keys, else the only variable in the file."""
    mat = sio.loadmat(path)
    for key in preferred_keys:
        if key in mat:
            return np.asarray(mat[key])
    variables = [k for k in mat if not k.startswith("__")]
    if len(variables) != 1:
        raise KeyError(f"{path}: cannot guess the variable among {variables}; expected one of {preferred_keys}")
    return np.asarray(mat[variables[0]])


def normalize_channels(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    minimum = features.min(axis=(0, 1), keepdims=True)
    shifted = features - minimum
    maximum = shifted.max(axis=(0, 1), keepdims=True)
    return np.divide(shifted, maximum, out=np.zeros_like(shifted), where=maximum != 0)


def apply_pca(hsi: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    height, width, bands = hsi.shape
    flat = hsi.reshape(-1, bands).astype(np.float32)
    pca = PCA(n_components=n_components, random_state=0)
    reduced = pca.fit_transform(flat)
    print(f"PCA {bands} -> {n_components} bands, explained variance {pca.explained_variance_ratio_.sum():.4f}")
    return reduced.reshape(height, width, n_components).astype(np.float32)


def random_split(train_map: np.ndarray, test_map: np.ndarray, per_class: int, seed: int):
    overlap = (train_map > 0) & (test_map > 0)
    if overlap.any():
        raise ValueError(f"{overlap.sum()} pixels are labelled in both train and test maps")
    ground_truth = np.where(train_map > 0, train_map, test_map).astype(np.int64)
    rng = np.random.RandomState(seed)
    new_train = np.zeros_like(ground_truth)
    for cls in np.unique(ground_truth[ground_truth > 0]):
        rows, cols = np.nonzero(ground_truth == cls)
        if len(rows) <= per_class:
            raise ValueError(f"class {cls} has only {len(rows)} pixels")
        chosen = rng.choice(len(rows), per_class, replace=False)
        new_train[rows[chosen], cols[chosen]] = cls
    new_test = np.where(new_train > 0, 0, ground_truth)
    return new_train, new_test


def class_counts(label_map: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(label_map[label_map > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or args.data_dir
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hsi = load_mat_array(raw_dir / args.hsi_file, ("spectraldata", "HSI", "hsi", "data"))
    lidar = load_mat_array(raw_dir / args.lidar_file, ("LiDAR_data", "LiDAR", "lidar", "DSM", "data"))
    train_map = load_mat_array(raw_dir / args.train_file, ("train_labels", "TRLabel", "TR_map", "TR"))
    test_map = load_mat_array(raw_dir / args.test_file, ("test_labels", "TSLabel", "TE_map", "TE"))
    if lidar.ndim == 2:
        lidar = lidar[..., None]
    print(f"HSI {hsi.shape}, LiDAR {lidar.shape}, train px {int((train_map > 0).sum())}, "
          f"test px {int((test_map > 0).sum())}")

    if args.pca:
        hsi = apply_pca(hsi, args.pca)
    features = normalize_channels(np.concatenate([hsi.astype(np.float32), lidar.astype(np.float32)], axis=2))

    if args.split == "random":
        train_map, test_map = random_split(train_map, test_map, args.train_per_class, args.seed)

    train_patches, train_labels, _ = make_patches(features, train_map, args.window_size)
    test_patches, test_labels, _ = make_patches(features, test_map, args.window_size)

    np.save(out_dir / "train_patches.npy", train_patches)
    np.save(out_dir / "train_labels.npy", train_labels)
    np.save(out_dir / "test_patches.npy", test_patches)
    np.save(out_dir / "test_labels.npy", test_labels)
    meta = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "hsi_channels": int(hsi.shape[2]),
        "lidar_channels": int(lidar.shape[2]),
        "train_per_class": class_counts(train_map),
        "test_per_class": class_counts(test_map),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(train_labels)} training and {len(test_labels)} test patches "
          f"({features.shape[2]} channels: {hsi.shape[2]} HSI + {lidar.shape[2]} LiDAR) to {out_dir}")


if __name__ == "__main__":
    main()
