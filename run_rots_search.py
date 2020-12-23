from pipeline import Pipeline
import numpy as np
from model.word_vector import VectorNames

def get_base_config():
    return {
        "name": "test_config",
        "word_vector": {
            "word_vector": "paranmt",
            "refresh_cache": False
        },
        "weight_scheme": "usif",
        "similarity": {
            "similarity": "rots",
        },
        "vector_convertor": {
            "vocab_conv": "nothing"
        },
        "sentence_parser": {
            "n_comp": 5,
            "scale": True
        },
        "dataset": {
            "dataset_name": "STSBenchmark",
            "task_name": "dev",
            "refresh_cache": False,
            "tokenizer": "spacy",
            "remove_stop": False,
            "remove_punc": True
        }
    }

if __name__ == "__main__":
    # change the word vectors
    for wv in VectorNames:

        log_file = "log/ROT_Search_{}.csv".format(wv)
        log = open(log_file, 'wt')
        log.write("coef_C,coef_P,left,mid,right\n")

        base_config = get_base_config()
        base_config['word_vector']['word_vector'] = wv
        for coef_C in np.linspace(0, 1, 11).tolist():
            for coef_P in np.logspace(-1, 3, 10, endpoint=False).tolist() + np.linspace(1000, 10000, num=10).tolist():
                base_config['similarity']['coef_C'] = coef_C
                base_config['similarity']['coef_P'] = coef_P
                p = Pipeline(base_config)
                p.preprocess()
                print("wv = {}, coef_C = {}, coef_P = {}".format(wv, coef_C, coef_P))
                left, mid, right = p.run()
                log.write("{},{},{},{},{}\n".format(coef_C, coef_P, left, mid, right))
                log.flush()

        log.close()


