from pipeline import Pipeline
from pipeline import get_config as _get_config
import numpy as np
from model.word_vector import VectorNames


def get_config(preprocess):
    cfg = _get_config("config/ROTS+{}.yaml".format(preprocess))
    cfg['similarity']['similarity'] = 'rots'
    cfg['dataset']['dataset_name'] = 'stsb:dev'
    return cfg



if __name__ == "__main__":
    # change the word vectors
    for pre in ['SUP', 'SWC', 'WR']:
        for wv in ['fasttext', 'glove840b', 'word2vec']:
            log_file = "log/hyperparameter_dev_{}_{}.csv".format(pre, wv)
            log = open(log_file, 'wt')
            log.write("dataset,correction,prior,left,mid,right\n")

            base_config = get_config(pre)
            base_config['word_vector']['word_vector'] = wv
            for dataset in ['stsb:test', 'twitter:test', 'sick:r']:
                base_config['dataset']['dataset_name'] = dataset
                for coef_C in np.linspace(0, 1, 5).tolist():
                    for coef_P in np.linspace(1, 16, 6):
                        base_config['similarity']['coef_C'] = coef_C
                        # constant
                        base_config['similarity']['preg'] = [coef_P
                            for _ in range(base_config['similarity']['depth'])]
                        p = Pipeline(base_config)
                        p.preprocess()
                        print("wv = {}, coef_C = {}, coef_P = {}".format(wv, coef_C, coef_P))
                        ans = p.run()
                        left, mid, right = ans['4']
                        log.write("{},{},{},{},{},{}\n".format(dataset, coef_C, coef_P, left, mid, right))
                        log.flush()
            log.close()


