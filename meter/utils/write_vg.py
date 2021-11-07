import json
import pandas as pd
import pyarrow as pa
import random
import os
import json

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]
    widths = [c["width"] for c in cdicts]
    heights = [c["height"] for c in cdicts]
    xs = [c["x"] for c in cdicts]
    ys = [c["y"] for c in cdicts]

    return [
        binary,
        captions,
        widths,
        heights,
        xs,
        ys,
        str(iid),
    ]


def make_arrow(root, dataset_root):
    imgs = {}
    with open(f"{root}/vg.train.img.tsv", "r") as fp:
        imgs['train'] = fp.readlines()
    captions = {}
    with open(f"{root}/vg.train.caption.tsv", "r") as fp:
        captions['train'] = fp.readlines()
    def to_batch(images, texts):
        img_lines = [l.strip().split('\t') for l in images]
        imgid2image = {}
        for imgid, img in img_lines:
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
            batches.append([imgid2image[imgid], imgid2image[imgid], imgid])
        return batches 

    for split in ["train"]:
        batches = to_batch(imgs[split], captions[split]) 
        print(len(batches))

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/vg.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
