import json
import os
import pandas as pd
import pyarrow as pa
import random
import json
import base64

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(root, dataset_root):
    imgs = {}
    with open(f"{root}/coco.train.img.tsv", "r") as fp:
        imgs['train'] = fp.readlines()
    with open(f"{root}/coco.val.img.tsv", "r") as fp:
        imgs['val'] = fp.readlines()
    with open(f"{root}/coco.test.img.tsv", "r") as fp:
        imgs['test'] = fp.readlines()
    captions = {}
    with open(f"{root}/coco.train.caption.tsv", "r") as fp:
        captions['train'] = fp.readlines()
    with open(f"{root}/coco.val.caption.tsv", "r") as fp:
        captions['val'] = fp.readlines()
    with open(f"{root}/coco.test.caption.tsv", "r") as fp:
        captions['test'] = fp.readlines()
    def to_batch(images, texts):
        img_lines = [l.strip().split('\t') for l in images]
        imgid2image = {}
        for imgid, img in img_lines:
            img = base64.b64decode(img)
            imgid2image[imgid] = img
        
        txt_lines = [l.strip().split('\t') for l in texts]
        imgid2text = {}
        for imgid, txt in txt_lines:
            txt = json.loads(txt)
            txt_list = [l['caption'] for l in txt]
            imgid2text[imgid] = txt_list

        assert len(imgid2image) == len(imgid2text)
        batches = []
        for imgid in imgid2image:
            batches.append([imgid2image[imgid], imgid2text[imgid], imgid])
        return batches 

    for split in ["train", "val", "test"]:
        batches = to_batch(imgs[split], captions[split]) 
        print(len(batches))
        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/coco_caption_karpathy_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
