import json
import os
from utils import read_from_yaml

from tqdm import tqdm
from pprint import pprint
from model import (get_dataset,
                   get_similarity, get_vector_convertor, get_weight_scheme,
                   get_word_vector, get_sentence_parser)
from scipy.stats import pearsonr, spearmanr
import scikits.bootstrap as boot


def get_config(config_file):
    if isinstance(config_file, dict):
        return config_file
    else:
        if config_file.endswith('json'):
            with open(config_file, 'rt') as f:
                config = json.load(f)
        elif config_file.endswith('yaml'):
            config = read_from_yaml(config_file)
        else:
            raise NotImplementedError
        return config


class Pipeline:
    def __init__(self, pipeline_config):
        """
        config can be a dict or a string that indicates the config file
        """
        self.config = get_config(pipeline_config)
        # pprint(self.config)
        # get word vectors accordingly
        self.word_vector = get_word_vector(**self.config['word_vector'])
        self.word_weight = get_weight_scheme(self.config['weight_scheme'])
        self.similarity = get_similarity(**self.config['similarity'])
        self.vocab_convertor = get_vector_convertor(**self.config['vector_convertor'])
        self.sentence_parser = get_sentence_parser(**self.config['sentence_parser'])

        self.dataset = get_dataset(**self.config['dataset'])
        self.name = self.config['name']

    def _evaluate_over_datasets(self):
        """Run evaluation for certain dataset
        """
        fail_cases = 0
        y_true = []
        y_pred = []
        for s1, s2, yt in tqdm(self.dataset.pairs):
            sent1 = self.sentence_parser.get_sentence(token_list=self.dataset.sentences[s1],
                                                      word_vector=self.word_vector,
                                                      weight_scheme=self.word_weight,
                                                      dataset=self.dataset)
            sent2 = self.sentence_parser.get_sentence(token_list=self.dataset.sentences[s2],
                                                      word_vector=self.word_vector,
                                                      weight_scheme=self.word_weight,
                                                      dataset=self.dataset)

            try:
                yp = self.similarity(sent1, sent2)
            # assert yp < 1
                y_true.append(yt)
                y_pred.append(yp)
            except:
                fail_cases += 1
        print('fail cases', fail_cases)
        if isinstance(y_pred[0], dict):
            y_pred_dict = {}
            for i in y_pred[0]:
                y_pred_dict[str(i)] = [yp[i] for yp in y_pred]
            return y_true, y_pred_dict
        else:
            return y_true, y_pred

    def inference(self):
        y_pred = []
        id_list = []
        for s1, s2, i in tqdm(self.dataset.pairs):
            sent1 = self.sentence_parser.get_sentence(token_list=self.dataset.sentences[s1],
                                                      word_vector=self.word_vector,
                                                      weight_scheme=self.word_weight,
                                                      dataset=self.dataset)
            sent2 = self.sentence_parser.get_sentence(token_list=self.dataset.sentences[s2],
                                                      word_vector=self.word_vector,
                                                      weight_scheme=self.word_weight,
                                                      dataset=self.dataset)

            yp = self.similarity(sent1, sent2)
            # assert yp < 1
            y_pred.append(yp)
            id_list.append(i)

        if isinstance(y_pred[0], dict):
            y_pred_dict = {}
            for i in y_pred[0]:
                y_pred_dict[str(i)] = [yp[i] for yp in y_pred]
            return y_pred_dict, id_list
        else:
            return y_pred, id_list

    def _evaluation_score(self, y_true:list, y_pred:list, option:str=""):
        if option == 'spearman':
            scoring = spearmanr
        else:
            scoring = pearsonr

        if isinstance(y_pred, list):
            score = scoring(y_true, y_pred)[0]
            left, right = boot.ci((y_true, y_pred), statfunction=lambda x, y: scoring(x, y)[0])
            return left, score, right
        elif isinstance(y_pred, dict):
            ans = {}
            for k in y_pred:
                l, s, r = self._evaluation_score(y_true, y_pred[k], option=option)
                ans[k] = [l, s, r]
            return ans

    def preprocess(self):
        # check if already processed, if so, just load the results, if not redo the processing
        # A word_vector contains
        word_vector_path = "tmp/{}/{}.word_vector".format(self.dataset.name, self.word_vector.name)
        if os.path.exists(word_vector_path) and self.config['word_vector'].get('refresh_cache', False) == False:
            self.word_vector.load_from_file(word_vector_path)
        else:
            os.makedirs(os.path.dirname(word_vector_path), exist_ok=True)
            self.word_vector.load(word2id=self.dataset.word_dict_rev)
            self.word_vector.save_to_file(word_vector_path)

        # self.word_weight.load(self.dataset.word_dict_rev)
        if self.vocab_convertor:
            self.vocab_convertor.update(self.word_vector)
        self.sentence_parser.update(self.word_vector, self.word_weight, self.dataset)
        print("[pipeline][preprocess] filtered out words without pretrained vectors")

    def run(self, pred=False, option='p'):
        yt, yp = self._evaluate_over_datasets()
        ans = self._evaluation_score(yt, yp, option=option)
        if isinstance(ans, tuple):
            l, s, r = ans
            print("{:.2f}, {:.2f}, {:.2f}".format(l*100, s*100, r*100))
        if isinstance(ans, dict):
            for k, (l, s, r) in ans.items():
                print("{}, {:.2f} & [{:.2f}, {:.2f}]".format(k, s*100, l*100, r*100))
        if pred:
            return ans, yt, yp
        else:
            return ans


if __name__ == "__main__":
    p = Pipeline(pipeline_config='config/default.yaml')
    p.preprocess()
    ans, yt, yp = p.run(pred=True)
    with open('test-COS.json', 'wt') as f:
        json.dump({
            'target': yt, 'predict': yp
        }, f)