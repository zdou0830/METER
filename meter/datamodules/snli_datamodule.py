from ..datasets import SNLIDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class SNLIDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SNLIDataset

    @property
    def dataset_name(self):
        return "snli"
