from .base_dataset import BaseDataset


class SNLIDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["snli_train"]
        elif split == "val":
            names = ["snli_dev", "snli_test"]
        elif split == "test":
            names = ["snli_dev", "snli_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="sentences",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]

        labels = self.table["labels"][index][question_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "labels": labels,
            "table_name": self.table_names[index],
        }
