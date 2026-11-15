"""Berlin HSI/SAR dataset loader."""

from __future__ import annotations

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .patches import make_patches


def _normalize_channels(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    minimum = features.min(axis=(0, 1), keepdims=True)
    shifted = features - minimum
    maximum = shifted.max(axis=(0, 1), keepdims=True)
    return np.divide(shifted, maximum, out=np.zeros_like(shifted), where=maximum != 0)


class HsiSarData(Dataset):
    """Create class-filtered HSI/SAR patches from the Berlin `.mat` files."""

    def __init__(
        self,
        root_berlin_hs,
        root_berlin_sar,
        root_train,
        root_test,
        classes=range(8),
        train=True,
        window_size: int = 9,
    ):
        super().__init__()
        self.train = train

        hsi = sio.loadmat(root_berlin_hs)["data_HS_LR"]
        sar = sio.loadmat(root_berlin_sar)["data_SAR_HR"]
        label_key = "TrainImage" if train else "TestImage"
        label_path = root_train if train else root_test
        labels = sio.loadmat(label_path)[label_key]

        features = _normalize_channels(np.concatenate([hsi, sar], axis=2))
        patches, patch_labels, _ = make_patches(features, labels, window_size)
        patch_labels = patch_labels.astype(np.int64) - 1
        mask = np.isin(patch_labels, np.asarray(tuple(classes)))

        data = patches[mask].transpose(0, 3, 1, 2)
        selected_labels = patch_labels[mask]
        if train:
            self.train_data = data
            self.train_labels = selected_labels
        else:
            self.test_data = data
            self.test_labels = selected_labels

    @property
    def data(self) -> np.ndarray:
        return self.train_data if self.train else self.test_data

    @property
    def labels(self) -> np.ndarray:
        return self.train_labels if self.train else self.test_labels

    def __getitem__(self, index):
        image = torch.as_tensor(self.data[index], dtype=torch.float32)
        target = torch.as_tensor(self.labels[index], dtype=torch.long)
        return index, image, target

    def __len__(self):
        return len(self.data)

    def get_image_class(self, label):
        if not self.train:
            raise RuntimeError("get_image_class is only available for the training split")
        return self.train_data[self.train_labels == label]
