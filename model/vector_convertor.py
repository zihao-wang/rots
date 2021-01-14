from .word_vector import Word2Vector
import numpy as np
from sklearn.decomposition import TruncatedSVD

def get_vector_convertor(vec_conv, **kwargs):
    seq = vec_conv.split('-')
    print(seq)
    convertor_seq = []
    for op in seq:
        if op == 'abtt':
            convertor_seq.append(AllButTheTop())
        elif op == 'cn':
            convertor_seq.append(ConceptorNegation())
        elif op == 'cent':
            convertor_seq.append(Centralize())
        elif op == 'norm':
            convertor_seq.append(Normalize())
    return Convertors(conv_seq=convertor_seq)


class Convertors:
    def __init__(self, conv_seq):
        self.conv_seq = conv_seq

    def update(self, w2v):
        for c in self.conv_seq:
            c.update(w2v)


class AllButTheTop:
    def __init__(self, D=3, **kwargs):
        self.D = D
        self.svd = TruncatedSVD(n_components=D, random_state=0)
        self.transfer = lambda x: x

    def update(self, w2v: Word2Vector):
        # compute mean
        vocab_matrix = np.asarray([w2v[w] for w in w2v.vectors])
        n, d = vocab_matrix.shape
        mean = np.mean(vocab_matrix, axis=0)
        vocab_matrix -= mean
        self.svd.fit(vocab_matrix)
        A = np.eye(d)
        for i in range(self.D):
            v = self.svd.components_[i].reshape(1, -1)
            A -= np.dot(v.T, v)
        self.transfer = lambda x: np.dot(A, (x.reshape(-1) - mean.reshape(-1))).reshape(-1)
        for w in w2v.vectors:
            w2v.vectors[w] = self.transfer(w2v[w])

class Centralize:
    def __init__(self):
        pass

    def update(self, w2v):
        vec_mat = np.asarray([w2v.vectors[k] for k in w2v.vectors])
        mean = np.mean(vec_mat, axis=0)
        for k in w2v.vectors:
            w2v.vectors[k] -= mean

class Normalize:
    def __init__(self):
        pass

    def update(self, w2v):
        for k in w2v.vectors:
            w2v.vectors[k] /= np.linalg.norm(w2v.vectors[k])


class ConceptorNegation:
    def __init__(self, alpha=2, **kwargs):
        self.alpha = alpha
        self.transfer = lambda x: x

    def update(self, w2v: Word2Vector):
        # compute mean
        vocab_matrix = np.asarray([w2v[w] for w in w2v.vectors])
        n, d = vocab_matrix.shape
        R = 0
        for i in range(n):
            w = vocab_matrix[i].reshape(-1, 1)
            R += np.dot(w, w.T)
        R /= n
        C = np.eye(d) - np.dot(R, np.linalg.inv(R + np.eye(d) * self.alpha ** -2))
        self.transfer = lambda x: np.dot(C, x.reshape(-1, 1)).reshape(-1)
        for w in w2v.vectors:
            w2v.vectors[w] = self.transfer(w2v[w])
