from .word_vector import Word2Vector
import numpy as np
from sklearn.decomposition import TruncatedSVD

def get_vector_convertor(vocab_conv, **kwargs):
    if vocab_conv == 'all_but_the_top':
        return AllButTheTop(**kwargs)
    elif vocab_conv == 'conceptor_negation':
        return ConceptorNegation(**kwargs)
    else:
        return  None

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
