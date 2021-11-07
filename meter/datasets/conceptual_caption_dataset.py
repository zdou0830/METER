from glob import glob
from .base_dataset import BaseDataset
import io
from PIL import Image


class ConceptualCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"conceptual_caption_train_{i}" for i in range(31)]
        elif split == "val":
            names = []

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")


    def __getitem__(self, index):
        return self.get_suite(index)

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }
