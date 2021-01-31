from pipeline import Pipeline
from pipeline import get_config as _get_config
import numpy as np
from model.word_vector import VectorNames

models = {"COS+SUP": "config/COS+SUP.yaml"}

def get_config(preprocess):
    cfg = _get_config("config/COS+{}.yaml".format(preprocess))
    cfg['similarity']['similarity'] = 'wrdinterp'
    cfg['dataset']['dataset_name'] = 'stsb:dev'
    return cfg



if __name__ == "__main__":
    # change the word vectors
    for pre in ['SUP', 'SWC', 'WR']:
        for wv in VectorNames:
            log_file = "log/Interpolation_dev_{}_{}.csv".format(pre, wv)
            log = open(log_file, 'wt')
            log.write("coef_C,coef_P,left,mid,right\n")

            base_config = get_config(pre)
            base_config['word_vector']['word_vector'] = wv
            for coef_C in np.linspace(0, 1, 11).tolist():
                for coef_P in [0] + np.logspace(-1, 3, 10, endpoint=False).tolist():
                    base_config['similarity']['coef_C'] = coef_C
                    base_config['similarity']['coef_P'] = coef_P
                    p = Pipeline(base_config)
                    p.preprocess()
                    print("wv = {}, coef_C = {}, coef_P = {}".format(wv, coef_C, coef_P))
                    left, mid, right = p.run()
                    log.write("{},{},{},{},{}\n".format(coef_C, coef_P, left, mid, right))
                    log.flush()

            log.close()


