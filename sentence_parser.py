import re
from copy import copy
import spacy
import numpy as np

from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


class processor:
    def __init__(self, vec):
        self.spacy_model = spacy.load('en_core_web_sm')
        self.vec = vec

    def preprocess(self, sentence):
        not_punc = re.compile('.*[A-Za-z0-9].*')

        # preprocess a given token
        def preprocess(t):
            t = t.lower().strip("';.:()").strip('"')
            t = 'not' if t == "n't" else t
            return re.split(r'[-]', t)

        tokens = []

        for token in nlp(sentence):
            if not_punc.match(token.text):
                tokens = tokens + preprocess(token.text)

        tokens = list(filter(lambda t: t in self.vec, tokens))



class Weights:
    """Map words to their probabilities."""
    def __init__(self, count_fn):
        """Initialize a word2prob object.

        Args:
            count_fn: word count file name (one word per line)
        """
        self.prob = {}
        total = 0.0
        with open(count_fn, encoding='utf8') as f:
            for line in tqdm(f.readlines(), desc='load {count_fn}'):
                k, v = line.split()
                v = int(v)
                k = k.lower()
                self.prob[k] = v
                total += v

        self.prob = {k: self.prob[k] / total for k in self.prob }
        self.min_prob = min(self.prob.values())
        self.count = total

    def __getitem__(self, w):
        return self.prob.get(w.lower(), self.min_prob)

    def __contains__(self, w):
        return w.lower() in self.prob

    def __len__(self):
        return len(self.prob)

    def vocab(self):
        return iter(self.prob.keys())


class uSIFWeight:
    def __init__(self, prob, n=11):
        self.prob = prob
        vocab_size = float(len(prob))
        threshold = 1 - (1-1/vocab_size) ** n
        alpha = len([w for w in prob.vocab() if prob[w] > threshold]) / vocab_size
        Z = 0.5 * vocab_size
        self.a = (1-alpha)/(alpha * Z)
        self.weight = lambda word: self.a / (0.5 * self.a + self.prob[word])

    def __getitem__(self, word):
        return self.weight(word)


class Word2Vector:
    def __init__(self, vector_fn, weight):
        """Initialize a word2vec object.

        Args:
            vector_fn: embedding file name (one word per line)
        """
        self.vectors = {}
        self.a = weight.a
        def _float(val):
            if val == '.':
                return float(0)
            else:
                return float(val)
        with open(vector_fn, mode='r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), desc='load word emb'):
                line = line.split()

                # skip first line if needed
                if len(line) == 2:
                    continue

                word = line[0]
                embedding = np.array([_float(val) for val in line[1:]])
                self.vectors[word] = embedding

    def __getitem__(self, w):
        word = w.lower()
        if word in self.vectors:
            return self.vectors[word]
        else:
            return np.zeros(300) + self.a

    def __contains__(self, w):
        return w in self.vectors


def get_uSIFw(count_fn):
    return uSIFWeight(Weights(count_fn))


class TreeNode:
    def __init__(self, v=None, w=None, token=None):
        self._v = v
        self._w = w
        self.token = token
        self.children = []
        # the center representation of belonging nodes
        self.v = None
        self.w = None

    def add_child(self, node):
        self.children.append(node)

    def parse(self):
        # if parsed
        # if self.w:
        #     return self.v, self.w

        # parsing
        if self._w:
            self.v, self.w = copy(self._v * self._w), copy(self._w)
        else:
            self.v, self.w = 0, 0

        for c in self.children:
            v, w = c.parse()
            if w:
                self.v += v * w
                self.w += w

        self.w = float(self.w)
        # make sure the vector * weights is the representation of nodes
        assert self.w > 0
        self.v /= self.w
        return self.v, self.w

    def get_children(self):
        if self._w:
            node = TreeNode(self._v, self._w, self.token)
            node.parse()
            return self.children + [node]
        else:
            return self.children

    def add_w_vec(self, vec):
        self._v += vec * self._w
        for c in self.children:
            c.add_w_vec(vec)


class Sentence:
    def __init__(self, vectors, weights, doc):
        self.doc = doc
        self.weights = weights
        vectors = np.array(vectors)
        self.vectors = vectors / np.linalg.norm(vectors, axis=0)
        self.dep_tree = None
        self.bin_tree = None

    def get_node(self, token):
        i = token.i
        return TreeNode(self.vectors[i].copy(), float(self.weights[i]), token)

    def parse_dep_tree(self):
        def _construct_spacy(token):
            node = self.get_node(token)
            # iterating spacy parsed children
            for child_token in token.children:
                node.add_child(_construct_spacy(child_token))
            return node

        #  if parsed
        if self.dep_tree:
            return self.dep_tree
        #  if not parsed
        root_nodes = [_construct_spacy(token) for token in self.doc if token.head == token]
        # if there is multiple root node
        if len(root_nodes) == 1:
            node = root_nodes[0]
        else:
            node = TreeNode()
            for n in root_nodes:
                node.add_child(n)
        node.parse()
        self.dep_tree = node
        return self.dep_tree

    def parse_bin_tree(self):
        def _construct_binary(seq):
            l = len(seq)
            if l >= 2:
                node = TreeNode()
                node.add_child(_construct_binary(seq[:l // 2]))
                node.add_child(_construct_binary(seq[l // 2:]))
            elif l == 1:
                node = self.get_node(seq[0])
            else:
                raise AssertionError("Structure error")
            return node

        #  if parsed
        if self.bin_tree:
            return self.bin_tree

        #  if not parsed
        self.bin_tree = _construct_binary(self.doc)
        self.bin_tree.parse()
        return self.bin_tree

    def get_sent_vec(self):
        top_tree_node = self.parse_bin_tree()
        return top_tree_node.w * top_tree_node.v

    def get_sent_vec_check(self):
        v = 0
        for i in range(len(self.weights)):
            v += self.weights[i] * self.vectors[i]
        v1 = self.get_sent_vec()
        another_tree_node = self.parse_dep_tree()
        v2 = another_tree_node.w * another_tree_node.v
        print("check sent vec: ", np.linalg.norm(v-v1), np.linalg.norm(v1-v2))
        return v

    def substract_vec(self, vec):
        self.vectors = [v - vec / sum(self.weights) for v in self.vectors]
        self.bin_tree = None
        self.dep_tree = None


class SentenceParser:
    def __init__(self, vector, weight, nlp):
        self.nlp = nlp
        self.weight = weight
        self.vector = vector
        self.sentence_list = []

    def add(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        if isinstance(sentences, list):
            for s in sentences:
                doc = self.nlp(s)
                weight = np.asarray([self.weight[t.text] for t in doc])
                weight /= len(weight)
                vector = [self.vector[t.text] for t in doc]
                self.sentence_list.append(Sentence(vector, weight, doc))

    def emb(self, m=5):
        vectors = []
        for s in self.sentence_list:
            vectors.append(s.get_sent_vec())

        proj = lambda a, b: a.dot(b.transpose()) * b
        svd = TruncatedSVD(n_components=m, random_state=0).fit(vectors)

        for i in range(m):
            lambda_i = (svd.singular_values_[i] ** 2) / (svd.singular_values_ ** 2).sum()
            pci = svd.components_[i]
            for sent in self.sentence_list:
                sent.substract_vec(lambda_i * proj(sent.get_sent_vec(), pci))

    def refresh(self):
        self.sentence_list = []

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        return self.sentence_list[item]


def get_model(count_fn="enwiki_vocab_min200.txt", vector_fn="vectors/psl.txt", **kwargs):
    print(count_fn, vector_fn)
    weight = get_uSIFw(count_fn)
    nlp = spacy.load('en_core_web_lg')
    vector = Word2Vector(vector_fn, weight)
    model = SentenceParser(vector, weight, nlp)
    return model
