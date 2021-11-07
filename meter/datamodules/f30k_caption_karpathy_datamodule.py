from ..datasets import F30KCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class F30KCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return F30KCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return F30KCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "f30k"

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
