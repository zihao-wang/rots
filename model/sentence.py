from copy import copy
from collections import defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
import spacy
# from benepar.spacy_plugin import BeneparComponent
from time import time

nlp = spacy.load('en_core_web_sm')

def get_sentence_parser(n_comp, scale, **kwargs):
    return GeneralPCompRemoval(n_comp, scale)


class Sentence:
    def __init__(self, vectors, weights, words, **kwargs):
        self.vectors = vectors
        self.weights = weights
        self.words = words
        assert len(self.words) == len(self.vectors) == len(self.words)
        self.tree_level_index = defaultdict(list)
        self.tree_level_span_flat = defaultdict(list)
        # each span is triple (begin, end, parent index)
        self.span_dict = {}  # sid : span
        self.span_vw_dict = {}  # sid : [v*w, w]
        self.span_begin_index = {} # store the span of each word, which is the element of the tree

    @property
    def sentence_vector(self):
        sent_vector = 0
        for v, w in zip(self.vectors, self.weights):
            sent_vector += v * w
        if len(self.vectors) == 0:
            return np.ones((1, 300))
        else:
            return sent_vector

    @property
    def string(self):
        return " ".join(self.words)

    def __len__(self):
        return len(self.weights)

    def _span_register(self, span):
        sid = len(self.span_dict)
        self.span_dict[sid] = span
        return sid

    def span_vw(self, span):
        begin, end, _ = span
        chunk_vw = 0
        chunk_wv = 0
        for i in range(begin, end):
            chunk_vw += self.vectors[i] * self.weights[i]
            chunk_wv += self.weights[i] * np.linalg.norm(self.vectors[i])
        return chunk_vw, chunk_wv

    def span_str(self, span):
        begin, end, _ = span
        return " ".join(self.words[begin: end])

    def parse(self, parser):
        # refresh the tree index
        self.tree_level_index = defaultdict(list)
        self.tree_level_span_flat = defaultdict(list)
        self.span_dict = {}
        self.span_vw_dict = {}
        self.span_begin_index = {}
        t1 = time()
        if parser.lower() in 'dependency':
            self._parse_spacy()
        elif parser.lower() in 'constituency':
            # this is very slow, so I don't recommnd to use this.
            # meanwhile, it produces similari results as dependency parser
            self.bc = BeneparComponent('benepar_en_small')
            if not nlp.has_pipe(self.bc):
                nlp.add_pipe(self.bc)
            self._parse_spacy()
        else:
            self._parse_binary()
        parse_time = time() - t1
        self.update_tree_vw()
        return parse_time

    def _parse_spacy(self):
        doc = nlp(self.string)
        # FIXME: caution if this assertion is false
        assert len(doc) == len(self.words)

        roots = [t for t in doc if t.head == t]
        def partition(token, level, sid):
            """
            Args:
                token: the token represented by the span, which maintains the connection information of the span
                level: the level of the span
                sid: the sid of parent span
            Return:
                begin: the begin of this token
                end: the end of this token
            """
            # target #1 construct the span for this token
            token_span = (None, None, sid) # initialize a dummy span to secure the sid
            begin = token.i
            end = token.i + 1
            token_sid = self._span_register(token_span)
            for child_token in token.children:
                b, e = partition(child_token, level+1, token_sid)
                begin = min(begin, b)
                end = max(end, e)
            token_span = (begin, end, sid)
            self.span_dict[token_sid] = token_span
            self.tree_level_index[level].append(token_span)

            if end - begin > 1:
                word_span = (token.i, token.i+1, token_sid)
                self._span_register(word_span)
                self.tree_level_index[level+1].append(word_span)
            return begin, end

        if len(roots) == 1:
            partition(roots[0], 0, None)
        else:
            root_span = (0, len(self), None)
            rsid = self._span_register(root_span)
            self.tree_level_index[0].append(root_span)
            for token in roots:
                partition(token, 1, rsid)

        for l in self.tree_level_index:
            self.tree_level_index[l] = sorted(self.tree_level_index[l], key=lambda x: x[0])

    def _parse_binary(self):
        def partition(span, level, sid):
            begin, end, _ = span
            # if can be further divided
            if end - begin > 1:
                mid = (begin + end) // 2
                assert begin < mid < end

                sub_span1 = (begin, mid, sid)
                ssid1 = self._span_register(sub_span1)
                self.tree_level_index[level].append(sub_span1)
                partition(sub_span1, level + 1, ssid1)

                sub_span2 = (mid, end, sid)
                ssid2 = self._span_register(sub_span2)
                self.tree_level_index[level].append(sub_span2)
                partition(sub_span2, level + 1, ssid2)

        root_span = (0, len(self), None)
        rsid = self._span_register(root_span)
        self.tree_level_index[0] = [root_span]
        partition(root_span, 1, rsid)

    def update_tree_vw(self):
        if len(self.span_dict) == 0:
            return
        d = len(self.tree_level_index) - 1
        while d >= 0:
            for span in self.tree_level_index[d]:
                b, e, p = span
                if e - b == 1:  # if is single word
                    vw, wvnorm = self.span_vw(span)
                    self.span_vw_dict[span] = [vw, wvnorm]
                    self.span_begin_index[b] = span
                else:
                    vw, wvnorm = self.span_vw_dict[span]

                if p in self.span_dict:
                    p_span = self.span_dict[p]
                    if p_span in self.span_vw_dict:
                        _vw, _wvnorm = self.span_vw_dict[p_span]
                        self.span_vw_dict[p_span] = [_vw + vw, _wvnorm + wvnorm]
                    else:
                        self.span_vw_dict[p_span] = [vw, wvnorm]
            d -= 1
        return

    def get_level_vectors_weights(self, l):
        # l = min(len(self.tree_level_index)-1, l)
        vectors, wvnorms = [], []
        b = 0
        li = 0
        while b < len(self):
            if li < len(self.tree_level_index[l]) and b == self.tree_level_index[l][li][0]:
                get_level_span = self.tree_level_index[l][li]
                _vw, _wvnorm = self.span_vw_dict[get_level_span]
                vectors.append(_vw)
                wvnorms.append(_wvnorm)
                b = get_level_span[1]
                li += 1
                self.tree_level_span_flat[l].append(get_level_span)
            else:
                get_word_span = self.span_begin_index[b]
                _vw, _wvnorm = self.span_vw_dict[get_word_span]
                vectors.append(_vw)
                wvnorms.append(_wvnorm)
                b += 1
                self.tree_level_span_flat[l].append(get_word_span)

        if (l-1) in self.tree_level_span_flat:
            top_down_link = defaultdict(list)
            down_span_flat = self.tree_level_span_flat[l]
            j = 0
            for i, tsp in enumerate(self.tree_level_span_flat[l-1]):
                tb, te, _ = tsp
                db, de, _ = down_span_flat[j]
                while tb <= db and de <= te:
                    top_down_link[i].append(j)
                    j += 1
                    if j >= len(down_span_flat): break
                    db, de, _ = down_span_flat[j]
        else:
            top_down_link = {}

        return vectors, wvnorms, top_down_link



def proj(x, pc):
    return x.dot(pc.transpose()) * pc


class GeneralPCompRemoval:
    def __init__(self, n_comp=0, scale=True, centralize=False, **sentence_args):
        self.n_comp = n_comp
        self.p_comp_i = []
        self.lambda_i = []
        self.transfer = lambda x: x
        self.scale_flag = scale
        self.centralize_flag = centralize
        self.sentence_args = sentence_args

    def scaling(self, vector_list):
        if self.scale_flag:
            vectors = np.array(vector_list)
            vectors = vectors / (np.linalg.norm(vectors, axis=0, keepdims=True) + 1e-10)
            return [vectors[i] for i in range(len(vector_list))]
        else:
            return vector_list

    def centralize(self, vector_list):
        if self.centralize_flag:
            vectors = np.array(vector_list)
            vectors = vectors - np.mean(vectors, axis=0)
            return [vectors[i] for i in range(len(vector_list))]
        else:
            return vector_list

    def update(self, word_vector, weight_scheme, dataset):

        sent_vec_list = []
        reduced_sentences = []
        for i, sent in enumerate(dataset.sentences):
            tokens = []
            words = []
            for t in sent:
                if dataset.word_dict[t] in word_vector:
                    tokens.append(t)
                    words.append(dataset.word_dict[t])
            reduced_sentences.append(tokens)



            sent_vec = np.zeros(300)
            word_vector_list = self.scaling([word_vector[w] for w in words])
            word_vector_list = self.centralize(word_vector_list)
            for i, w in enumerate(words):
                sent_vec += word_vector_list[i] * weight_scheme[w] / len(words)
            sent_vec_list.append(sent_vec)

        dataset.sentences = reduced_sentences

        if self.n_comp > 0:
            sent_vectors = np.asarray(sent_vec_list)
            svd = TruncatedSVD(n_components=self.n_comp)
            svd.fit(sent_vectors)

            for i in range(self.n_comp):
                self.lambda_i.append(
                    (svd.singular_values_[i] ** 2) / np.sum(svd.singular_values_ ** 2))
                self.p_comp_i.append(svd.components_[i])

    def get_sentence(self, token_list, word_vector, weight_scheme, dataset):
        words = [dataset.word_dict[t] for t in token_list if dataset.word_dict[t] in word_vector]
        assert len(words) == len(token_list)
        vectors = self.scaling([word_vector[w] for w in words])
        vectors = self.centralize(vectors)
        weights = [weight_scheme[w] / len(words) for w in words]
        sum_weights = sum(weights)
        for i in range(self.n_comp):
            vectors = [v - proj(v, self.p_comp_i[i]) * self.lambda_i[i] * w / sum_weights
                       for v, w in zip(vectors, weights)]
        return Sentence(vectors, weights, words, **self.sentence_args)


if __name__ == "__main__":
    string = 'the answers so far are already good but i d like to add a map for switzerland'

    from dataset import not_punc, preprocess
    _tokens = [t.text for t in nlp(string)]
    words = []
    for t in _tokens:
        if not_punc.match(t): words.extend(preprocess(t))
    print(words)
    L = len(words)
    d = 300
    vectors = np.random.randn(L, d)
    weights = np.ones(L) / L
    sent = Sentence(vectors, weights, words)
    parsers = ['dependency', 'binary', 'constituency']
    for p in parsers:
        t = sent.parse(parser=p)
        print(p, t)
        print(sent.tree_level_index)
        for l in sent.tree_level_index:
            vectors, weights, *_ = sent.get_level_vectors_weights(l)
            v = 0
            for _v, _w in zip(vectors, weights):
                v += _v * _w

            assert np.linalg.norm(v - sent.sentence_vector) < 1e-10