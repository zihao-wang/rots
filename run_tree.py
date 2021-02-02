from pipeline import Pipeline
from pipeline import get_config as _get_config
import numpy as np
from model.word_vector import VectorNames


def get_config(preprocess):
    cfg = _get_config("config/ROTS+{}.yaml".format(preprocess))
    cfg['similarity']['similarity'] = 'rots'
    cfg['similarity']['parser'] = 'bin'
    return cfg



if __name__ == "__main__":
    # change the word vectors
    log_file = "log/binary_parser.csv"
    log = open(log_file, 'wt')
    log.write("dataset,pre,vector,similarity,left,mid,right\n")
    for pre in ['SUP', 'SWC', 'WR']:
        for wv in ['fasttext', 'glove840b', 'word2vec']:
            base_config = get_config(pre)
            base_config['word_vector']['word_vector'] = wv
            for dataset in ['stsb:test', 'twitter:test', 'sick:r']:
                base_config['dataset']['dataset_name'] = dataset
                p = Pipeline(base_config)
                p.preprocess()
                ans = p.run()
                for k in ans:
                    left, mid, right = ans[k]
                    log.write("{},{},{},{},{},{},{}\n".format(dataset, pre, wv, k, left, mid, right))
                log.flush()
    log.close()


