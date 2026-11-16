"""Class-incremental run on Houston 2013 / MUUFL / Augsburg with OA / AA / Kappa / APD per session.

Presets
-------
original : behaviour of scripts/train_incremental.py (7+3+3+2 classes, permuted
           class order, base m=6 then m=K//s, Adam lr 1e-4 batch 16 everywhere,
           optimizer created once).
paper    : setting described in BiCM-Prompt (IEEE TIP 2026), Sec. IV / Table I:
           6+3+3+3 classes in the original order, memory K=30 split evenly over
           the seen classes, base session Adam lr 1e-3 batch 32, optimizer
           rebuilt for the parameters trainable in each session.
           Data must be prepared with PCA 40 + 180 random samples/class
           (see scripts/prepare_houston.py).

Every value can be overridden on the command line; values the paper does not
state (epochs, incremental lr/batch, loss weights) keep the code defaults.

    python -m scripts.run_cil --preset paper --data-dir data_houston/paper_seed0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from my_cil.data import HsiLiDARData
from my_cil.evaluation import calculate_forgetting_rate, convert_to_stage_stats
from my_cil.training import MyCilTrainNet

PRESETS = {
    "original": dict(base_classes=7, increment=3, total_classes=15, permute=True, memory=30,
                     memory_rule="original", base_epochs=30, inc_epochs=35, base_lr=1e-4, inc_lr=1e-4,
                     base_batch=16, inc_batch=16, fix_optimizer=False),
    "paper": dict(base_classes=6, increment=3, total_classes=15, permute=False, memory=30,
                  memory_rule="even", base_epochs=30, inc_epochs=35, base_lr=1e-3, inc_lr=1e-4,
                  base_batch=32, inc_batch=16, fix_optimizer=True),
}

# Class split / memory per dataset (paper Tables I-III, Sec. IV-B: K chosen so the
# last session stores 2 / 3 / 5 exemplars per class). Only used with --preset paper.
DATASETS = {
    "houston": dict(base_classes=6, increment=3, total_classes=15, memory=30),
    "muufl": dict(base_classes=5, increment=2, total_classes=11, memory=33),
    "augsburg": dict(base_classes=3, increment=2, total_classes=7, memory=35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, default="paper")
    parser.add_argument("--dataset", choices=DATASETS, default="houston")
    parser.add_argument("--data-dir", type=Path, default=Path("data_houston"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="JSON result file (default: results/<preset>_seed<seed>.json)")
    parser.add_argument("--num-workers", type=int, default=4)
    on_off = argparse.BooleanOptionalAction
    parser.add_argument("--base-classes", type=int)
    parser.add_argument("--increment", type=int)
    parser.add_argument("--total-classes", type=int)
    parser.add_argument("--permute", action=on_off, help="permute class order (seed 4) like the original code")
    parser.add_argument("--memory", type=int, help="total exemplar memory K")
    parser.add_argument("--memory-rule", choices=("original", "even"))
    parser.add_argument("--base-epochs", type=int)
    parser.add_argument("--inc-epochs", type=int)
    parser.add_argument("--base-lr", type=float)
    parser.add_argument("--inc-lr", type=float)
    parser.add_argument("--base-batch", type=int)
    parser.add_argument("--inc-batch", type=int)
    parser.add_argument("--fix-optimizer", action=on_off)
    args = parser.parse_args()
    if args.preset == "paper":
        for key, value in DATASETS[args.dataset].items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(data_dir: Path, classes, train: bool, permute: bool) -> HsiLiDARData:
    return HsiLiDARData(
        root_train_data=data_dir / "train_patches.npy",
        root_train_label=data_dir / "train_labels.npy",
        root_test_data=data_dir / "test_patches.npy",
        root_test_label=data_dir / "test_labels.npy",
        train=train,
        classes=classes,
        permute_labels=permute,
    )


@torch.no_grad()
def predict(model, loader, hsi_channels: int, use_nme: bool, device):
    model.eval()
    preds, labels = [], []
    for _, images, target in loader:
        images = images.to(device)
        x1, x2 = images[:, :hsi_channels], images[:, hsi_channels:]
        if use_nme:
            out = model.classify(x1, x2)
        else:
            out = model(x1, x2)[0].argmax(dim=1)
        preds.append(out.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    classes = np.unique(labels)
    per_class = {int(c): float((preds[labels == c] == c).mean()) for c in classes}
    return {
        "OA": float((preds == labels).mean() * 100),
        "AA": float(np.mean(list(per_class.values())) * 100),
        "Kappa": float(cohen_kappa_score(labels, preds) * 100),
        "per_class_acc": per_class,
        "confusion": confusion_matrix(labels, preds).tolist(),
    }


def exemplars_per_class(args, n_seen: int, n_old: int, stage: int) -> int:
    if args.memory_rule == "even":
        return args.memory // n_seen
    return 6 if stage == 0 else args.memory // n_old  # original script: A=6, then m = K // s


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = args.data_dir / "meta.json"
    sample = np.load(args.data_dir / "train_patches.npy", mmap_mode="r")
    total_channels = sample.shape[-1]
    hsi_channels = json.loads(meta_path.read_text())["hsi_channels"] if meta_path.exists() else total_channels - 1
    aux_channels = total_channels - hsi_channels
    print(f"data: {args.data_dir} channels={total_channels} (HSI {hsi_channels}, aux {aux_channels})")
    print("config:", {k: str(v) for k, v in vars(args).items()})

    model = MyCilTrainNet(num_group_tokens=3, group_feature_size=128, feature_size=128, num_classes=0,
                          hsi_channels=hsi_channels, batch_size=args.base_batch, init_lr=args.inc_lr,
                          base_lr=args.base_lr, fix_optimizer=args.fix_optimizer,
                          aux_channels=aux_channels)

    boundaries = [args.base_classes]
    while boundaries[-1] < args.total_classes:
        boundaries.append(min(boundaries[-1] + args.increment, args.total_classes))

    sessions, correct_list, total_list = [], [], []
    start_all = time.time()
    n_old = 0
    for stage, n_seen in enumerate(boundaries):
        start = time.time()
        new_classes = range(n_old, n_seen)
        print(f"\n===== session {stage}: classes {list(new_classes)} (seen {n_seen}) =====")
        train_set = make_dataset(args.data_dir, new_classes, True, args.permute)
        model.batch_size = args.base_batch if stage == 0 else args.inc_batch
        model.update_representation(train_set, args.base_epochs if stage == 0 else args.inc_epochs)
        model.eval()

        m = exemplars_per_class(args, n_seen, n_old, stage)
        print("exemplars per class m =", m)
        model.reduce_exemplar_set(m)
        with torch.no_grad():
            for y in range(model.n_known, model.n_classes):
                model.construct_exemplar_set(train_set.get_image_class(y), m)
        model.n_known = model.n_classes

        test_all = make_dataset(args.data_dir, range(n_seen), False, args.permute)
        loader = torch.utils.data.DataLoader(test_all, batch_size=64, shuffle=False, num_workers=args.num_workers)
        # Evaluation protocol of the original script: NME classifier after the base
        # session, fc logits afterwards. Both are recorded.
        model.compute_means = True
        nme_preds, labels = predict(model, loader, hsi_channels, True, device)
        fc_preds, _ = predict(model, loader, hsi_channels, False, device)
        primary = nme_preds if stage == 0 else fc_preds
        result = {"session": stage, "seen_classes": n_seen, "m": m, "seconds": round(time.time() - start, 1),
                  **metrics(primary, labels),
                  "fc": metrics(fc_preds, labels), "nme": metrics(nme_preds, labels)}
        for extra in ("fc", "nme"):
            for key in ("confusion", "per_class_acc"):
                result[extra].pop(key)

        correct_list.append({c: int(((primary == c) & (labels == c)).sum()) for c in np.unique(labels).tolist()})
        total_list.append({c: int((labels == c).sum()) for c in np.unique(labels).tolist()})
        if stage > 0:
            result["APD"] = calculate_forgetting_rate(convert_to_stage_stats(correct_list, total_list))[-1]
        sessions.append(result)
        print(f"session {stage}: OA {result['OA']:.2f} AA {result['AA']:.2f} Kappa {result['Kappa']:.2f}"
              + (f" APD {result['APD']:.4f}" if stage else "")
              + f" | fc OA {result['fc']['OA']:.2f} nme OA {result['nme']['OA']:.2f} | {result['seconds']}s")
        torch.cuda.empty_cache()
        n_old = n_seen

    summary = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "hsi_channels": hsi_channels,
        "aux_channels": aux_channels,
        "sessions": sessions,
        "average": {k: float(np.mean([s[k] for s in sessions])) for k in ("OA", "AA", "Kappa")},
        "total_seconds": round(time.time() - start_all, 1),
    }
    output = args.output or Path("results") / (f"{args.preset}_seed{args.seed}.json" if args.dataset == "houston"
                                             else f"{args.dataset}_{args.preset}_seed{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print("\nsession  OA     AA     Kappa  APD")
    for s in sessions:
        print(f"{s['session']:>7}  {s['OA']:.2f}  {s['AA']:.2f}  {s['Kappa']:.2f}  {s.get('APD', float('nan')):.4f}")
    print("average  " + "  ".join(f"{summary['average'][k]:.2f}" for k in ("OA", "AA", "Kappa")))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
