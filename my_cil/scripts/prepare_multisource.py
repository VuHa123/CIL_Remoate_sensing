"""Prepare MUUFL (HSI + LiDAR) and Augsburg (HSI + SAR) like the BiCM-Prompt paper.

Paper (IEEE TIP 2026) Sec. IV-A/B, Tables II-III: 11x11 patches, HSI reduced to
40 PCA components, LiDAR/SAR channels kept, random per-class training split
(MUUFL 100/class, Augsburg 150/class), all remaining labelled pixels are test.

Sources
-------
muufl    : GatorSense/MUUFLGulfport, MUUFLGulfportSceneLabels/
           muufl_gulfport_campus_1_hsi_220_label.mat (HSI 325x220x64, labels 1..11,
           -1 = unlabelled). LiDAR: first raster of hsi.Lidar{1}.z (the paper
           uses a single LiDAR channel).
augsburg : songyz2019/rs-fusion-datasets-dist v1.0.0 augsburg-ouc.zip
           (augsburg_hsi 332x485x180, augsburg_sar 332x485x4, augsburg_gt 1..7).
           Labels are reordered to the paper's Table III order
           (Water, Commercial, Allotment, Low plants, Industrial, Residential, Forest),
           which is the reverse of the file order.

    python -m scripts.prepare_multisource --dataset muufl --raw-dir data_muufl/raw \
        --out-dir data_muufl/paper_seed0 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from my_cil.data import make_patches
from scripts.prepare_houston import apply_pca, class_counts, normalize_channels

TRAIN_PER_CLASS = {"muufl": 100, "augsburg": 150}


def load_muufl(raw_dir: Path):
    hsi = sio.loadmat(raw_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                      squeeze_me=True, struct_as_record=False)["hsi"]
    labels = np.asarray(hsi.sceneLabels.labels).astype(np.int64)
    labels[labels < 0] = 0
    lidar = np.asarray(hsi.Lidar[0].z, dtype=np.float32)[..., :1]
    return np.asarray(hsi.Data, dtype=np.float32), lidar, labels


def load_augsburg(raw_dir: Path):
    folder = raw_dir / "Augsburg"
    hsi = sio.loadmat(folder / "augsburg_hsi.mat")["augsburg_hsi"].astype(np.float32)
    sar = sio.loadmat(folder / "augsburg_sar.mat")["augsburg_sar"].astype(np.float32)
    gt = sio.loadmat(folder / "augsburg_gt.mat")["augsburg_gt"].astype(np.int64)
    gt = np.where(gt > 0, 8 - gt, 0)  # file order 1..7 -> paper Table III order
    return hsi, sar, gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=TRAIN_PER_CLASS, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pca", type=int, default=40)
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--window-size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    per_class = args.train_per_class or TRAIN_PER_CLASS[args.dataset]

    hsi, aux, gt = (load_muufl if args.dataset == "muufl" else load_augsburg)(args.raw_dir)
    print(f"HSI {hsi.shape}, aux {aux.shape}, labelled {int((gt > 0).sum())}")
    if args.pca:
        hsi = apply_pca(hsi, args.pca)
    features = normalize_channels(np.concatenate([hsi, aux], axis=2))

    rng = np.random.RandomState(args.seed)
    train_map = np.zeros_like(gt)
    for cls in np.unique(gt[gt > 0]):
        rows, cols = np.nonzero(gt == cls)
        chosen = rng.choice(len(rows), per_class, replace=False)
        train_map[rows[chosen], cols[chosen]] = cls
    test_map = np.where(train_map > 0, 0, gt)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, label_map in (("train", train_map), ("test", test_map)):
        patches, labels, _ = make_patches(features, label_map, args.window_size)
        np.save(args.out_dir / f"{name}_patches.npy", patches)
        np.save(args.out_dir / f"{name}_labels.npy", labels)
        print(f"{name}: {len(labels)} patches, {patches.shape[1:]}")
    meta = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "hsi_channels": int(hsi.shape[2]),
        "aux_channels": int(aux.shape[2]),
        "train_per_class": class_counts(train_map),
        "test_per_class": class_counts(test_map),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
