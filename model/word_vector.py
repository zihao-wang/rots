from tqdm import tqdm
import numpy as np


VectorNames = {
    "paragram": "vectors/paragram_300_sl999/paragram_300_sl999.txt",
    "fasttext": "vectors/crawl-300d-2M.vec",
    "fasttext_sw": "vectors/crawl-300d-2M-subword.vec",
    "paramnt": "vectors/czeng.txt",
    "glove": "vectors/glove.42B.300d.txt",
    "psl": "vectors/psl.txt"
}


def get_word_vector(word_vector, **kwargs):
    assert word_vector in VectorNames
    w2v = Word2Vector(VectorNames[word_vector])
    return w2v


class Word2Vector:
    def __init__(self, vector_fn, max_number=-1):
        """Initialize a word2vec object.

        Args:
            vector_fn: embedding file name (one word per line)
        """
        self.vectors = {}
        self.vector_fn = vector_fn
        self.max_number = max_number

    def load(self, restrict_words=None):
        def _float(val):
            if val == '.':
                return float(0)
            else:
                return float(val)

        print("[Word2Vector] load word vector data under restriction of " +
            "None" if len(restrict_words) == 0 else "set of {} words".format(len(restrict_words)))

        with open(self.vector_fn, mode='r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()

                # skip first line if needed
                if len(line) == 2:
                    continue

                word = line[0]
                if restrict_words and word not in restrict_words:
                    continue
                embedding = np.array([_float(val) for val in line[1:]])
                self.vectors[word] = embedding
                if 0 < self.max_number == len(self.vectors):
                    break

    def __getitem__(self, w):
        word = w.lower()
        return self.vectors.get(word, np.zeros(300))

    def __contains__(self, w):
        return w in self.vectors

    def update(self, word, vector):
        self.vectors[word] = vector


if __name__ == "__main__":
    from time import time
    for name in VectorNames:
        print(name, VectorNames[name])
        t1 = time()
        w2v = Word2Vector(vector_fn=VectorNames[name])
        w2v.load(restrict_words={'hi'})
        t2 = time()
        print("use time {}".format(t2-t1))
