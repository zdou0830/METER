import json
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm
from collections import defaultdict


label2id = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
def process(root, imgid, ann):
    with open(f"{root}/Flickr30K/images/{imgid}.jpg", "rb") as fp:
        img = fp.read()

    sentences = ann['sentences']

    labels = ann['labels']

    return [img, sentences, labels]



def make_arrow(root, dataset_root):
    train_data = list(
        map(json.loads, open(f"{root}/snli_ve_train.jsonl").readlines())
    )
    test_data = list(
        map(json.loads, open(f"{root}/snli_ve_test.jsonl").readlines())
    )
    dev_data = list(
        map(json.loads, open(f"{root}/snli_ve_dev.jsonl").readlines())
    )


    splits = [
        "train",
        "dev",
        "test",
    ]


    annotations = dict()
    annotations['train'] = train_data
    annotations['dev'] = dev_data
    annotations['test'] = test_data
    annots = dict()
    for split in splits:
        annots[split] = {}
        for line in annotations[split]:
            imgid = line['Flickr30K_ID']
            if not imgid in annots[split]:
                annots[split][imgid] = {}
                annots[split][imgid]['sentences'] = []
                annots[split][imgid]['labels'] = []
            annots[split][imgid]['sentences'].append( [line['sentence1'], line['sentence2']] )
            annots[split][imgid]['labels'].append( label2id[line['gold_label']] )
            
        

    for split in splits:
        bs = [process(root, imgid, annots[split][imgid]) for imgid in tqdm(annots[split])]

        dataframe = pd.DataFrame(
            bs, columns=["image", "sentences", "labels"]
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/snli_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
