"""Evaluate a trained continual-learning checkpoint on Houston test patches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader

from my_cil.data import HsiLiDARData
from my_cil.training import MyCilTrainNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data_houston"))
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device, num_classes: int) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, torch.nn.Module):
        return payload.to(device)
    model = MyCilTrainNet(3, 128, 128, num_classes).to(device)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HsiLiDARData(
        root_train_data=args.data_dir / "train_patches.npy",
        root_train_label=args.data_dir / "train_labels.npy",
        root_test_data=args.data_dir / "test_patches.npy",
        root_test_label=args.data_dir / "test_labels.npy",
        train=False,
        classes=range(args.num_classes),
    )
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_model(args.checkpoint, device, args.num_classes)
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for _, images, labels in loader:
            images = images.to(device)
            logits = model(images[:, :144], images[:, 144:])[0]
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    matrix = confusion_matrix(all_labels, all_predictions)
    class_accuracies = np.divide(
        matrix.diagonal(),
        matrix.sum(axis=1),
        out=np.zeros(matrix.shape[0], dtype=float),
        where=matrix.sum(axis=1) != 0,
    )
    print(f"Kappa: {cohen_kappa_score(all_labels, all_predictions):.4f}")
    print(f"Mean class accuracy: {class_accuracies.mean():.4f}")
    for class_id, accuracy in enumerate(class_accuracies):
        print(f"Class {class_id}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
