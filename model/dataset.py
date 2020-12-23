import pickle
import spacy
import os
import re
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def preprocess(t):
    t = t.lower().strip("';.:()").strip('"')
    t = 'not' if t == "n't" else t
    return re.split(r'[-]', t)

not_punc = re.compile('.*[A-Za-z0-9].*')


dataset_dict = {
    "STSBenchmark": "dataset/STSBenchmark"
}



class Dataset:
    """A dataset maintains:
        - dataset-wise dicts
            - word_dict: {id: word string}
            - word_dict_rev: {word string: id}

        - list of sentencs, for i-th sentence
            - indiced sentences [[word_ids]]
        - sentence pair (sent_id1, sent_id2, true_score)
    """

    def __init__(self, dataset, task_name, tokenizer='nltk', remove_stop=False, remove_punc=False):
        self.name = dataset
        if '/' in self.name:
            self.name = self.name.split('/')[-1]
        self.task = task_name
        if dataset in dataset_dict:
            self.dataset = dataset_dict[dataset]
        else:
            self.dataset = dataset
        self.tokenizer = tokenizer
        self.remove_stop = remove_stop
        self.remove_punc = remove_punc

        self.raw_sentences = []
        self.sentences = []
        self.scores = []

        self.word_dict = {}
        self.word_dict_rev = {}

        self.pairs = []

        # self.load()
    def load_from_file(self, path):
        print("[dataset] load dataset from {}".format(path))
        with open(path, 'rb') as f:
            self.raw_sentences, self.sentences, self.scores, self.word_dict, self.word_dict_rev, self.pairs = pickle.load(f)

    def save_to_file(self, path):
        print("[dataset] save dataset to {}".format(path))
        with open(path, 'wb') as f:
            pickle.dump([self.raw_sentences, self.sentences, self.scores, self.word_dict, self.word_dict_rev, self.pairs], f)

    def __len__(self):
        return len(self.pairs)

    def load(self):
        print('[dataset][load] load task: {} from dataset: {}'.format(self.task, self.dataset))
        if self.task in ['test', 'dev']:
            task = 'sts-{}.csv'.format(self.task)
        else:
            task = self.task
        with open(os.path.join(self.dataset, task), encoding='utf8', mode='rt') as f:
            for line in f.readlines():
                try:
                    similarity, s1, s2 = line.strip().split('\t')[-3:]
                    self.raw_sentences.extend([s1, s2])
                    self.scores.append(float(similarity))
                except:
                    pass

    def tokenization(self):
        print("[dataset][Tokenization]")
        if not self.raw_sentences:
            raise ValueError("No raw sentences loaded")
        token_count = 0
        split_count = 0
        if self.tokenizer == 'spacy':
            print("[dataset][Tokenization] Using spacy tokenizer")
            from .sentence import nlp as tokenizer
        else:
            print("[dataset][Tokenization] Using nltk tokenizer")
            from nltk import word_tokenize

            # preprocess a given token

        for sentence in tqdm(self.raw_sentences):
            if self.tokenizer == 'spacy':
                if self.remove_stop:
                    _tokens = [t.text for t in tokenizer(sentence) if t.is_stop == False]
                else:
                    _tokens = [t.text for t in tokenizer(sentence)]
            else:
                _tokens = [t for t in word_tokenize(sentence)]

            tokens = []
            for t in _tokens:
                if self.remove_punc and not not_punc.match(t):
                    continue
                tokens.extend(preprocess(t))

            tokens = [t for t in tokens if t]
            token_count += len(tokens)
            split_count += len(sentence.split())

            for token in tokens:
                if token not in self.word_dict_rev:
                    i = len(self.word_dict)
                    self.word_dict[i] = token
                    self.word_dict_rev[token] = i

            self.sentences.append([self.word_dict_rev[t] for t in tokens])

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


def get_dataset(dataset_name="STSBenchmark", task_name="test", refresh_cache=False, **kwargs):
    d = Dataset(dataset_name, task_name, **kwargs)
    dataset_path = "tmp/{}-{}.data".format(d.name, d.task)
    if os.path.exists(dataset_path) and refresh_cache == False:
        d.load_from_file(dataset_path)
    else:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        d.load()
        d.tokenization()
        d.pairing()
        d.save_to_file(dataset_path)
    return d


if __name__ == "__main__":
    dataset_name = "dataset/STS/STS-data/STS2014-gold"
    task_names = filter(lambda fn: '.input.' in fn and fn.endswith('txt'), os.listdir(dataset_name))
    for task_name in task_names:
        d = get_dataset(dataset_name, task_name)
        # print(d.pairs)

