import json
import numpy
import os
# import yaml

from tqdm import tqdm

from model import (get_dataset,
                   get_similarity, get_vector_convertor, get_weight_scheme,
                   get_word_vector, get_sentence_parser)
from scipy.stats import pearsonr
import scikits.bootstrap as boot

def get_config(config_file):
    if isinstance(config_file, dict):
        return config_file
    else:
        with open(config_file, 'rt') as f:
            if config_file.endswith('json'):
                config = json.load(f)
            else:
                raise NotImplementedError
        return config


class Pipeline:
    def __init__(self, pipeline_config:str):
        self.config = get_config(pipeline_config)

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
        y_true = []
        y_pred = []
        for s1, s2, yt in tqdm(self.dataset.pairs):
            y_true.append(yt)
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
        return y_true, y_pred

    def _evaluation_score(self, y_true:list, y_pred:list, option:str=""):
        score = pearsonr(y_true, y_pred)[0]
        left, right = boot.ci((y_true, y_pred), statfunction=lambda x, y: pearsonr(x, y)[0])
        return left, score, right

    def preprocess(self):
        # check if already processed, if so, just load the results, if not redo the processing
        # A word_vector contains
        word_vector_path = "tmp/{}-{}/{}.word_vector".format(self.dataset.name, self.dataset.task, self.word_vector.name)
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

    def run(self):
        yt, yp = self._evaluate_over_datasets()
        scores = self._evaluation_score(yt, yp)
        print(scores)
        return scores


if __name__ == "__main__":
    p = Pipeline(pipeline_config='config/test.json')
    p.preprocess()
    p.run()