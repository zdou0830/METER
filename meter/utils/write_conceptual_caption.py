import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
import json

from tqdm import tqdm
from glob import glob


def path2rest(path, iid2captions):
    split, _, name = path.split("/")[-3:]
    split = split.split("_")[-1]
    iid = name

    with open(path, "rb") as fp:
        binary = fp.read()

    captions = iid2captions[iid]

    return [
        binary,
        captions,
        iid,
        split,
    ]


def make_arrow(root, dataset_root):
    imgs = {}
    print('reading image...')
    cnt = 0
    with open(f"{root}//cc.train.img.split0{cnt}", "r") as fp:
        imgs['train'] = fp.readlines()
    captions = {}
    print('reading caption...')
    with open(f"{root}/cc.train.caption.tsv", "r") as fp:
        captions['train'] = fp.readlines()
    print('spliting...')
    def to_batch(images, texts, batch_len=100000):
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
        batchess = []
        batches = []
        for imgid in imgid2image:
            batches.append([imgid2image[imgid], imgid2image[imgid], imgid])
            if len(batches) == batch_len:
                batchess.append(batches)
                batches = []
        if len(batches) > 0:
            batchess.append(batches)
        return batchess 

    for split in ["train"]:
        batchess = to_batch(imgs[split], captions[split]) 
        for batch_id, batches in enumerate(batchess):
            print(len(batches))

            dataframe = pd.DataFrame(
                batches, columns=["image", "caption", "image_id"],
            )

            table = pa.Table.from_pandas(dataframe)
            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/conceptual_caption_{split}_{batch_id}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
