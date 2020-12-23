import pickle
import spacy
import os
import re


nlp = spacy.load('en_core_web_sm')


class Sentence:
    pass


class Dataset:
    """A dataset maintains:
        - dataset-wise dicts
            - word_dict: {id: word string}
            - word_dict_rev: {word string: id}

        - list of sentencs, for i-th sentence
            - indiced sentences [[word_ids]]
        - sentence pair (sent_id1, sent_id2, true_score)
    """

    def __init__(self, dataset):
        self.dataset = dataset

        self.raw_sentences = []
        self.sentences = []
        self.scores = []

        self.word_dict = {}
        self.word_dict_rev = {}

        self.pairs = []

        # self.load()

    def __len__(self):
        return len(self.pairs)

    def load(self, task_name):
        print('[dataset][load] load task: {} from dataset: {}'.format(task_name, self.dataset))
        if self.dataset == 'dataset/STSBenchmark':
            with open(os.path.join(self.dataset, 'sts-{}.csv'.format(task_name)), encoding='utf8', mode='rt') as f:
                for line in f.readlines():
                    similarity, s1, s2 = line.strip().split('\t')[-3:]
                    self.raw_sentences.extend([s1, s2])
                    self.scores.append(float(similarity))
        else:
            # todo this should input the filename, but it can be created by scripts
            with open(os.path.join(self.dataset, task_name), encoding='utf8', mode='rt') as f:
                self.raw_sentences = re.split(r'\t|\n', f.read().strip())
            with open(os.path.join(self.dataset, task_name.replace('input', 'gs')), encoding='utf8', mode='rt') as f:
                self.scores = list(map(float, f.read().strip().split('\n')))

    def tokenization(self):
        print("[dataset][Tokenization]")
        if not self.raw_sentences:
            raise ValueError("No raw sentences loaded")
        token_count = 0
        split_count = 0

        print("[dataset][Tokenization] Using spacy tokenizer")
        from spacy.tokenizer import Tokenizer
        tokenizer = Tokenizer(nlp.vocab)

        for sentence in self.raw_sentences:
            tokens = tokenizer(sentence)

            token_count += len(tokens)
            split_count += len(sentence.split())

            for t in tokens:
                token = t.text
                if token not in self.word_dict_rev:
                    i = len(self.word_dict)
                    self.word_dict[i] = token
                    self.word_dict_rev[token] = i

            self.sentences.append([self.word_dict_rev[t.text] for t in tokens])

        print("[dataset][Tokenization] Finished with\n" +
              "\t- token coverage over spacy blocks {}/{}, {}%\n".format(token_count, split_count,  100 * token_count/split_count) +
              "\t- total number of tokens {}\n".format(len(self.word_dict)) +
              "\t- total number of sentences {}".format(len(self.sentences)))

    def pairing(self):
        assert len(self.sentences) == len(self.raw_sentences)
        assert len(self.sentences) % 2 == 0
        print("[dataset][Pairing] Pairing cases")
        for i in range(0, len(self.sentences), 2):
            self.pairs.append([i, i+1, self.scores[i//2]])
        print("[dataset][Pairing] finished with {} pairs".format(len(self.pairs)))


def get_dataset(dataset_name="STSBenchmark", task_name="test", **kwargs):
    d = Dataset(dataset_name)
    d.load(task_name)
    d.tokenization()
    d.pairing()
    return d


if __name__ == "__main__":
    dataset_name = "dataset/STS/STS-data/STS2014-gold"
    task_names = filter(lambda fn: '.input.' in fn and fn.endswith('txt'), os.listdir(dataset_name))
    for task_name in task_names:
        d = get_dataset(dataset_name, task_name)
        # print(d.pairs)

