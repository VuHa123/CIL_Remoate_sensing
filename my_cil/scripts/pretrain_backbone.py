"""Pretrain one ViT branch on Houston HSI or LiDAR patches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from my_cil.data import HsiLiDARData
from my_cil.models import ViTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_houston"))
    parser.add_argument("--modality", choices=("hsi", "lidar"), default="hsi")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def create_dataset(data_dir: Path, *, train: bool, num_classes: int) -> HsiLiDARData:
    return HsiLiDARData(
        root_train_data=data_dir / "train_patches.npy",
        root_train_label=data_dir / "train_labels.npy",
        root_test_data=data_dir / "test_patches.npy",
        root_test_label=data_dir / "test_labels.npy",
        train=train,
        classes=range(num_classes),
    )


def select_modality(images: torch.Tensor, modality: str) -> torch.Tensor:
    return images[:, :144] if modality == "hsi" else images[:, 144:]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 144 if args.modality == "hsi" else 1
    output = args.output or Path(f"vit_model_{args.modality}.pth")

    train_loader = DataLoader(
        create_dataset(args.data_dir, train=True, num_classes=args.num_classes),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        create_dataset(args.data_dir, train=False, num_classes=args.num_classes),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ViTModel(11, 1, in_channels, 768, 8, 8, args.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for _, images, labels in train_loader:
            images = select_modality(images.float().to(device), args.modality)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for _, images, labels in test_loader:
                images = select_modality(images.float().to(device), args.modality)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"loss={running_loss / len(train_loader):.4f} accuracy={100 * correct / total:.2f}%"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
