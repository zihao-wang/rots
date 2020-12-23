from tqdm import tqdm
import numpy as np
import pickle


VectorNames = {
    "paragram": "vectors/paragram_300_sl999/paragram_300_sl999.txt",
    "fasttext": "vectors/crawl-300d-2M.vec",
    "fasttext_sw": "vectors/crawl-300d-2M-subword.vec",
    "paranmt": "vectors/czeng.txt",
    "glove840b": "vectors/glove.840B.300d.txt",
    "glove42b": "vectors/glove.840B.300d.txt",
    "psl": "vectors/psl.txt"
}


def get_word_vector(word_vector, **kwargs):
    name = word_vector.lower()
    assert name in VectorNames
    w2v = Word2Vector(name)
    return w2v


class Word2Vector:
    def __init__(self, vector_name, max_number=-1):
        """Initialize a word2vec object.

        Args:
            vector_fn: embedding file name (one word per line)
        """
        self.name = vector_name
        self.vectors = {}
        self.vector_fn = VectorNames[vector_name]
        self.max_number = max_number

    def save_to_file(self, path):
        print("[Word2Vector] save word vector to file {}".format(path))
        with open(path, 'wb') as f:
            pickle.dump(self.vectors, f)

    def load_from_file(self, path):
        print("[Word2Vector] load word vector from file {}".format(path))
        with open(path, 'rb') as f:
            self.vectors = pickle.load(f)

    def load(self, word2id=None):
        def _float(val):
            if val == '.':
                return float(0)
            else:
                return float(val)

        print("[Word2Vector] load word vector data under restriction of " +
            "None" if len(word2id) == 0 else "set of {} words".format(len(word2id)))
        with open(self.vector_fn, mode='r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                line = line.split()

                # skip first line if needed
                if len(line) == 2:
                    continue

                word = line[0].lower()
                if word2id and word not in word2id:
                    continue
                embedding_vals = []
                for val in line[1:]:
                    try:
                        f = _float(val)
                        embedding_vals.append(f)
                    except:
                        pass
                embedding = np.array(embedding_vals)
                self.vectors[word] = embedding[:300]
                if 0 < self.max_number == len(self.vectors):
                    break

    def __getitem__(self, w):
        return self.vectors.get(w.lower(), np.zeros(300))

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
        w2v.load(word2id={'hi'})
        t2 = time()
        print("use time {}".format(t2-t1))
