from .sentence import Sentence
import numpy as np
import ot
from scipy.spatial.distance import cdist


def cosine(x, y):
    def shape_norm(v):
        v = np.asarray(v)
        if len(v.shape) == 1:
            v = v.reshape(1, -1)
        return v
    x, y = shape_norm(x), shape_norm(y)
    n1, d1 = x.shape
    n2, d2 = y.shape
    assert d1 == d2
    M = 1 - (np.dot(x, y.T)) / (np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(y, axis=1, keepdims=True).T + 1e-10)
    return np.squeeze(M).reshape(n1, n2)


def get_similarity(similarity="cos", **kwargs):
    name = similarity.lower()
    if name == 'cos':
        return CosineSimilarity(**kwargs)
    elif name == 'wmd':
        return WMDp(**kwargs)
    elif name == 'wrd':
        return WRD(**kwargs)
    elif name == 'wrdlevels':
        return WRDLevels(**kwargs)
    elif name == 'wrdinterp':
        return WRDInterp(**kwargs)
    elif name == 'wfrrdinterp':
        return WFRRDInterp(**kwargs)
    elif name == 'rots':
        return ROTS(**kwargs)


class CosineSimilarity:
    def __init__(self, adjust_WRD=False, **kwargs):
        self.adjust_WRD = adjust_WRD

    def __call__(self, s1, s2):
        sim = 1 - cosine(s1.sentence_vector, s2.sentence_vector).squeeze()
        if self.adjust_WRD:
            sim *= np.linalg.norm(s1.sentence_vector)
            sim *= np.linalg.norm(s2.sentence_vector)
            a = np.asarray([np.linalg.norm(v) * w for v, w in zip(s1.vectors, s1.weights)])
            sim /= np.sum(a)
            b = np.asarray([np.linalg.norm(v) * w for v, w in zip(s2.vectors, s2.weights)])
            sim /= np.sum(b)
        return sim


class WMDp:
    def __init__(self, p=2, **kwargs):
        self.p = p

    def __call__(self, s1, s2):
        M = cdist(np.asarray(s1.vectors),
                  np.asarray(s2.vectors), 'minkowski', p=self.p)
        a = np.asarray(s1.weights)
        a /= np.sum(a)
        b = np.asarray(s2.weights)
        b /= np.sum(b)
        return 1 - ot.emd2(a, b, M)


@DeprecationWarning
class WFRRDInterp:
    def __init__(self, coef_C, coef_P, radius, **kwargs):
        self.coef_C = coef_C
        self.coef_P = coef_P
        self.radius = radius

    def __call__(self, s1, s2):
        _M = cosine(s1.vectors, s2.vectors)
        M = - np.log(np.cos(np.minimum(_M / self.radius * np.pi/2, np.pi/2 - 1e-3) + 1e-3))
        _a = np.asarray([np.linalg.norm(v) * w for v, w in zip(s1.vectors, s1.weights)])
        a = _a / np.sum(_a)
        _b = np.asarray([np.linalg.norm(v) * w for v, w in zip(s2.vectors, s2.weights)])
        b = _b / np.sum(_b)
        C = np.sum(_a) * np.sum(_b) / (np.linalg.norm(s1.sentence_vector) * np.linalg.norm(s2.sentence_vector) + 1e-3)
        prior = a.reshape(-1, 1).dot(b.reshape(1, -1))
        M_prior = _M - self.coef_P * np.log(prior)
        P = ot.unbalanced.sinkhorn_unbalanced(a, b, M_prior, reg=self.coef_P, reg_m=10, method="sinkhorn_stabilized")
        ans = (1 - np.sum(P * _M)) * (1 - self.coef_C + self.coef_C * C)
        # assert ans < 1
        return ans


class WRD:
    def __init__(self, adjust_cos=False, **kwargs):
        self.adjust_cos = adjust_cos

    def __call__(self, s1, s2):
        if len(s1.vectors) == 0 or len(s2.vectors) == 0:
            return 1
        if len(s1.vectors) == 1 or len(s2.vectors) == 1:
            return CosineSimilarity()(s1, s2)
        M = cosine(np.asarray(s1.vectors), np.asarray(s2.vectors))
        _a = np.asarray([np.linalg.norm(v) * w for v, w in zip(s1.vectors, s1.weights)])
        a = _a / np.sum(_a)
        _b = np.asarray([np.linalg.norm(v) * w for v, w in zip(s2.vectors, s2.weights)])
        b = _b / np.sum(_b)
        sim = 1 - ot.emd2(a, b, M)
        if self.adjust_cos:
            sim *= np.sum(_a) * np.sum(_b)
            sim /= np.linalg.norm(s1.sentence_vector)
            sim /= np.linalg.norm(s2.sentence_vector)
        return sim

class WRDLevels:
    def __init__(self, depth=5, margin='norm_vectors', parser='dep', **kwargs):
        """
        margin: one of norm_vectors or vector_norms
        - norm_vectors: norm of cumulated vectors
        - vector_norms: cumulated vector norms
        """
        self.margin = margin
        self.depth = depth
        self.parser = parser

    def __call__(self, s1, s2):
        if len(s1.vectors) == 0 or len(s2.vectors) == 0:
            answer = {d: 1 for d in range(self.depth)}
        elif len(s1.vectors) == 1 or len(s2.vectors) == 1:
            answer = {d: float(CosineSimilarity()(s1, s2)) for d in range(self.depth)}
        else:
            s1.parse(self.parser)
            s2.parse(self.parser)
            answer = {}
            for d in range(self.depth):
                vectors1, wvnorms1, _ = s1.get_level_vectors_weights(d)
                vectors2, wvnorms2, _ = s2.get_level_vectors_weights(d)
                if self.margin == 'norm_vectors':
                    _a = np.asarray([np.linalg.norm(v) for v in vectors1])
                    _b = np.asarray([np.linalg.norm(v) for v in vectors2])
                elif self.margin == 'vector_norms':
                    _a = np.asarray(wvnorms1)
                    _b = np.asarray(wvnorms2)
                else:
                    raise NotImplementedError
                a = _a / np.sum(_a)
                b = _b / np.sum(_b)
                M = cosine(np.asarray(vectors1), np.asarray(vectors2))
                answer[d] = 1 - ot.emd2(a, b, M)

        return answer


class WRDInterp:
    def __init__(self, coef_C, coef_P, **kwargs):
        self.coef_C = coef_C
        self.coef_P = coef_P

    def __call__(self, s1, s2):
        M = cosine(s1.vectors, s2.vectors)
        _a = np.asarray([np.linalg.norm(v) * w for v, w in zip(s1.vectors, s1.weights)])
        a = _a / np.sum(_a)
        _b = np.asarray([np.linalg.norm(v) * w for v, w in zip(s2.vectors, s2.weights)])
        b = _b / np.sum(_b)
        C = np.sum(_a) * np.sum(_b) / (np.linalg.norm(s1.sentence_vector) * np.linalg.norm(s2.sentence_vector) + 1e-3)
        prior = a.reshape(-1, 1).dot(b.reshape(1, -1))
        if self.coef_P > 0:
            M_prior = M - self.coef_P * np.log(prior)
            P = ot.sinkhorn(a, b, M_prior, reg=self.coef_P, method="sinkhorn_stabilized")
        else:
            P = ot.emd(a, b, M)
        ans = (1 - np.sum(P * M)) * (1 - self.coef_C + self.coef_C * C)
        # assert ans < 1
        return ans


class ROTS:
    def __init__(self, parser='binary', depth=5, preg=2, creg=0, ereg=0, coef_C=1, aggregation='last', **kwargs):
        """
        Args:
            parser: type of parsers, in ['dependency', 'binary']
            depth: how deep you consider this
            preg: prior regularization
            creg: cosine regularization
            ereg: entropy regularization
            coef_C: C interpolation coefficient
            aggregation: how to handle different scores [mean, max, min, last, no]
        """
        self.parser = parser
        self.depth = depth
        if isinstance(preg, list):
            self.prior_reg = preg
        else:
            self.prior_reg = [preg * (i+1) for i in range(depth)]
            # self.prior_reg = [32 for i in range(depth)]
        self.creg = creg
        self.ereg = ereg
        self.coef_C = coef_C
        self.aggregation = aggregation

    def __call__(self, s1: Sentence, s2: Sentence):
        if len(s1.vectors) == 0 or len(s2.vectors) == 0:
            _depth = self.depth
            answer = {d: 1 for d in range(self.depth)}
        elif len(s1.vectors) == 1 or len(s2.vectors) == 1:
            answer = {d: float(CosineSimilarity()(s1, s2)) for d in range(self.depth)}
            _depth = self.depth
        else:
            s1.parse(self.parser)
            s2.parse(self.parser)
            _depth = min(max(len(s1.tree_level_index), len(s2.tree_level_index)), self.depth)
            depth = self.depth
            answer = {}  # d, alignment score
            transport_plan = {}
            for d in range(self.depth):
                # if d == 0:
                # vectors1, weights1, tdlink1 = s1.get_level_vectors_weights(d)
                # vectors2, weights2, tdlink2 = s2.get_level_vectors_weights(d)
                vectors1, _a, tdlink1 = s1.get_level_vectors_weights(d)
                vectors2, _b, tdlink2 = s2.get_level_vectors_weights(d)
                M_cossim = cosine(vectors1, vectors2)
                # _a = np.asarray([np.linalg.norm(v) * w for v, w in zip(vectors1, weights1)])
                a = _a / np.sum(_a)
                # _b = np.asarray([np.linalg.norm(v) * w for v, w in zip(vectors2, weights2)])
                b = _b / np.sum(_b)
                C = np.sum(_a) * np.sum(_b) / (np.linalg.norm(s1.sentence_vector) * np.linalg.norm(s2.sentence_vector) + 1e-3)
                cos_prior = a.reshape(-1, 1).dot(b.reshape(1, -1))
                if tdlink1 and tdlink2:
                    prior_plan_top = transport_plan[d-1]
                    prior_plan_down = np.copy(cos_prior)
                    for ti in tdlink1:
                        for tj in tdlink2:
                            mass = prior_plan_top[ti, tj]
                            local_a = np.sum([_a[di] for di in tdlink1[ti]])
                            local_b = np.sum([_b[dj] for dj in tdlink2[tj]])
                            for di in tdlink1[ti]:
                                for dj in tdlink2[tj]:
                                    prior_plan_down[di, dj] = mass * _a[di] * _b[dj] / local_a / local_b
                else:
                    # print(d)
                    prior_plan_down = cos_prior
                M = M_cossim - np.log(cos_prior + 1e-10) * self.creg - np.log(prior_plan_down + 1e-10) * self.prior_reg[d]
                reg = self.creg + self.prior_reg[d] + self.ereg
                P = ot.sinkhorn(a, b, M, reg, method='sinkhorn_stabilized',numItermax=32)
                # P = ot.emd(a, b, M)
                # P = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=reg, reg_m=1, method="sinkhorn_stabilized")
                transport_plan[d] = P
                coef_C = 0
                answer[d] = (1 - np.sum(P * M_cossim)) * (1 - coef_C + coef_C * C)
                # answer[d] = 1 - ot.emd2(a, b, M)

        if self.aggregation == 'mean':
            return np.mean(list(answer.values()))
        elif self.aggregation == 'max':
            return np.max(list(answer.values()))
        elif self.aggregation == 'min':
            return np.min(list(answer.values()))
        elif self.aggregation == 'last':
            return answer[_depth-1]
        elif self.aggregation == 'all':
            answer['mean'] = np.mean(list(answer.values()))
            answer['max'] = np.max(list(answer.values()))
            answer['min'] = np.min(list(answer.values()))
            answer['last'] = answer[_depth-1]
            return answer
        else:
            return answer




