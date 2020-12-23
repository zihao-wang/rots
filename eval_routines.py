import argparse
import os
import re

import numpy as np
import ot
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from model.similarities import cosine
from sentence_parser import Sentence, get_model

parser = argparse.ArgumentParser()
parser.add_argument('--hreg', default=10)
parser.add_argument('--kreg', default=0)
parser.add_argument('--ereg', default=0)
parser.add_argument('--tree', default='bin')
parser.add_argument('--depth', default=7)
parser.add_argument('--vector_fn', default='vectors/czeng.txt')
parser.add_argument('--WInterp', default=True)
parser.add_argument('--CInterp', default=True)
params = parser.parse_args()

def coss(s1: Sentence, s2: Sentence):
    v1 = s1.get_sent_vec()
    v2 = s2.get_sent_vec()
    return cosine(v1, v2)

def rots(s1: Sentence, s2: Sentence, depth=5, reg=(10, 0, 0), tree='bin', Winterp=True, Cinterp=True, **kwargs):
    """
    Recursive Optimal Transport Similarity
    Args:
        s1:
        s2:
        depth:
        reg:
        tree:
        Winterp:
        Cinterp:

    Returns: similarity

    """
    if tree == "dep":
        # print("dep tree")
        node1, node2 = s1.parse_dep_tree(), s2.parse_dep_tree()
    elif tree == "bin":
        # print("binary tree")
        node1, node2 = s1.parse_bin_tree(), s2.parse_bin_tree()
    else:
        raise AssertionError("tree not know")

    # on-the-fly bfs
    L1, L2 = [node1], [node2]
    hot_prior = 0
    ans = {}
    for d in range(depth+1):
        l1, l2 = len(L1), len(L2)
        w1, w2, v1, v2 = [], [], [], []
        nextL1, nextL2 = [], []
        # load data
        for n in L1:
            w1.append(n.w), v1.append(n.v.reshape((1, -1)))
            nextL1.append(n.get_children())
        for n in L2:
            w2.append(n.w), v2.append(n.v.reshape((1, -1)))
            nextL2.append(n.get_children())
        # weights and cost
        if Winterp:
            a = [_w * np.linalg.norm(_v) for _w, _v in zip(w1, v1)]
            b = [_w * np.linalg.norm(_v) for _w, _v in zip(w2, v2)]
        else:
            a = w1
            b = w2
        weight1, weight2 = np.asarray(a)/sum(a), np.asarray(b)/sum(b)
        v1mat = np.concatenate(v1, axis=0).reshape((l1, 1, -1))
        v2mat = np.concatenate(v2, axis=0).reshape((1, l2, -1))
        cost = 1 - np.sum(v1mat * v2mat, axis=-1)/np.linalg.norm(v1mat, axis=-1)/np.linalg.norm(v2mat, axis=-1)
        # coefficient and prior
        if Cinterp:
            C = (sum(a) * sum(b)) / \
                (np.linalg.norm(sum(_w * _v for _w, _v in zip(w1, v1))) *
                 np.linalg.norm(sum(_w * _v for _w, _v in zip(w2, v2))))
        else:
            C = 1
        prior = weight1.reshape([-1, 1]) * weight2.reshape([1, -1])
        kot_prior = - np.log(prior)
        if d > 1:
            c1 = 0
            hot_prior = np.copy(prior)
            for ii, nl1 in enumerate(LL1):
                c2 = 0
                for jj, nl2 in enumerate(LL2):
                    blk = hot_prior[c1: c1+len(nl1), c2: c2+len(nl2)]
                    hot_prior[c1: c1+len(nl1), c2: c2+len(nl2)] *= T[ii, jj] / np.sum(blk)
                    c2 += len(nl2)
                c1 += len(nl1)
            hot_prior = - np.log(hot_prior)
        else:
            hot_prior = 0

        T = ot.sinkhorn(weight1, weight2, cost + reg[0] * hot_prior + reg[1] * kot_prior, reg=sum(reg),
                        # numItermax=32,
                        method='sinkhorn_stabilized')
        sim = C * np.sum(T * (1-cost))
        ans['d' + str(d)] = sim
        # prepare next level
        L1 = []
        for nl in nextL1:
            L1 += nl
        L2 = []
        for nl in nextL2:
            L2 += nl
        LL1, LL2 = nextL1, nextL2
    return ans


def rots_result_parser(y, y_rots):
    eval_pairs = {'d-mean': [[], []]}
    for k in y_rots[0].keys():
        eval_pairs[k] = [[], []]
    assert len(y) == len(y_rots)
    for y, kv in zip(y, y_rots):
        dscores = []
        for k, v in kv.items():
            eval_pairs[k][0].append(y)
            eval_pairs[k][1].append(v)
            if 'd' in k and k != 'd0':
                dscores.append(v)
        eval_pairs['d-mean'][0].append(y)
        eval_pairs['d-mean'][1].append(np.mean(dscores))
    return eval_pairs


def show_eval_pairs(eval_pairs):
    data = {}
    for k, (_y, _y_hat) in eval_pairs.items():
        data[k] = [pearsonr(_y, _y_hat)[0]]
    df = pd.DataFrame(data)
    print(df.to_string())


def eval_STS_2012_2015(model, **kwargs):
    test_dirs = ['STS/STS-data/STS2012-gold/', 'STS/STS-data/STS2013-gold/',
                 'STS/STS-data/STS2014-gold/', 'STS/STS-data/STS2015-gold/']

    for td in test_dirs:
        test_fns = filter(lambda fn: '.input.' in fn and fn.endswith('txt'), os.listdir(td))
        print(td + "-" * 40)
        for fn in test_fns:
            # IO
            sentences = re.split(r'\t|\n', open(td + fn, encoding='utf8').read().strip())
            y = list(map(float, open(td + fn.replace('input', 'gs'), encoding='utf8').read().strip().split('\n')))
            # parse sentence
            model.refresh()
            model.add(sentences)
            model.emb()
            y_coss, y_rots = [], []
            for i in tqdm(range(0, len(sentences), 2)):
                y_coss.append(coss(model[i], model[i+1]))
                y_rots.append(rots(model[i], model[i+1], **kwargs))

            eval_pairs = rots_result_parser(y, y_rots)
            eval_pairs['uSIF'] = [y, y_coss]
            show_eval_pairs(eval_pairs)


def eval_STS_benchmark(model, **kwargs):
    test_fns = ['dataset/STSBenchmark/sts-dev.csv', 'dataset/STSBenchmark/sts-test.csv' ]

    for fn in test_fns:
        eval_pairs = {'d-mean': [[], []]}
        y, y_hat = [], []
        sentences = []

        for line in open(fn, encoding='utf8'):
            similarity, s1, s2 = line.strip().split('\t')[-3:]
            sentences.append(s1)
            sentences.append(s2)
            y.append(float(similarity))
        model.refresh()
        model.add(sentences)
        model.emb()
        y_coss, y_rots = [], []
        for i in tqdm(range(0, len(model), 2)):
            y_coss.append(coss(model[i], model[i+1]))
            y_rots.append(rots(model[i], model[i+1], **kwargs))

        eval_pairs = rots_result_parser(y, y_rots)
        eval_pairs['uSIF'] = [y, y_coss]
        show_eval_pairs(eval_pairs)


if __name__ == "__main__":
    kwargs = vars(params)
    kwargs['reg'] = [params.hreg, params.kreg, params.ereg]
    print(kwargs)
    model = get_model(**kwargs)
    # eval_STS_2012_2015(model, **kwargs)
    eval_STS_benchmark(model, **kwargs)
