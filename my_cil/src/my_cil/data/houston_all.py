"""Houston dataset variant that keeps the original class order."""

from .houston import NpyPatchDataset


class HsiLiDARDataAll(NpyPatchDataset):
    def __init__(
        self,
        root_train_data,
        root_train_label,
        root_test_data,
        root_test_label,
        train=True,
    ):
        super().__init__(
            root_train_data,
            root_train_label,
            root_test_data,
            root_test_label,
            classes=range(15),
            train=train,
            permute_labels=False,
        )


# Historical spelling retained for existing notebooks.
HsiLiDARData_all = HsiLiDARDataAll
