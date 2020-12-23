import os

import numpy as np
from tqdm import tqdm

from pipeline import Pipeline
from model.word_vector import VectorNames

sts_all = "/Users/zihao/Project/dataset-sts/data/sts/semeval-sts/all"
tasks = os.listdir(sts_all)

def get_base_config(task):
    return {
        "name": "test_config",
        "word_vector": {
            "word_vector": "paranmt",
            "refresh_cache": False
        },
        "weight_scheme": "usif",
        "similarity": {
            "similarity": "cos",
        },
        "vector_convertor": {
            "vocab_conv": "nothing"
        },
        "sentence_parser": {
            "n_comp": 5,
            "scale": True
        },
        "dataset": {
            "dataset_name": sts_all,
            "task_name": task,
            "refresh_cache": False,
            "tokenizer": "spacy",
            "remove_stop": False,
            "remove_punc": True
        }
    }

def run(cfg):
    p = Pipeline(cfg)
    p.preprocess()
    return p.run()

def index(cfg):
    p = Pipeline(cfg)
    p.preprocess()

if __name__ == "__main__":
    print(tasks)

    for t in tqdm(tasks):
        if t == '.DS_Store':
            continue
        year, task_name = t.split('.')[:2]
        cfg = get_base_config(t)
        for wv in VectorNames:
            print(wv)
            cfg['word_vector']['word_vector'] = wv
            index(cfg)


