"""Houston HSI/LiDAR NumPy patch datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyPatchDataset(Dataset):
    """Shared implementation for train/test patch arrays stored as `.npy`."""

    def __init__(
        self,
        root_train_data,
        root_train_label,
        root_test_data,
        root_test_label,
        *,
        classes,
        train: bool,
        permute_labels: bool,
        num_classes: int = 15,
        permutation_seed: int = 4,
    ):
        super().__init__()
        self.train = train
        data_path = root_train_data if train else root_test_data
        label_path = root_train_label if train else root_test_label

        data = np.load(data_path)
        labels = np.load(label_path).astype(np.int64) - 1
        if permute_labels:
            permutation = np.random.RandomState(permutation_seed).permutation(num_classes)
            labels = permutation[labels]
            self.label_perm = permutation

        mask = np.isin(labels, np.asarray(tuple(classes)))
        data = data[mask].transpose(0, 3, 1, 2)
        labels = labels[mask]
        if train:
            self.train_data = data
            self.train_labels = labels
        else:
            self.test_data = data
            self.test_labels = labels

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


class HsiLiDARData(NpyPatchDataset):
    """Houston split with the deterministic class permutation used by CIL runs."""

    def __init__(
        self,
        root_train_data,
        root_train_label,
        root_test_data,
        root_test_label,
        classes=range(15),
        train=True,
    ):
        super().__init__(
            root_train_data,
            root_train_label,
            root_test_data,
            root_test_label,
            classes=classes,
            train=train,
            permute_labels=True,
        )
