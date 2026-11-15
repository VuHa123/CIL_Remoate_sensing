"""Plot t-SNE embeddings produced by a trained continual learner."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from my_cil.data import HsiLiDARData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data_houston"))
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/tsne.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("plot_tsne expects a checkpoint containing the complete model object")
    model = model.to(device).eval()

    dataset = HsiLiDARData(
        root_train_data=args.data_dir / "train_patches.npy",
        root_train_label=args.data_dir / "train_labels.npy",
        root_test_data=args.data_dir / "test_patches.npy",
        root_test_label=args.data_dir / "test_labels.npy",
        train=False,
        classes=range(args.num_classes),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for _, images, batch_labels in loader:
            images = images.to(device)
            feature_mean = model(images[:, :144], images[:, 144:])[2]
            features.append(feature_mean.cpu().numpy())
            labels.append(batch_labels.numpy())

    embedded = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=100,
        random_state=20,
    ).fit_transform(np.concatenate(features))
    labels_array = np.concatenate(labels)

    figure = plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels_array, cmap="tab20")
    plt.colorbar(scatter, label="Labels")
    plt.title("Houston class-incremental features")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
