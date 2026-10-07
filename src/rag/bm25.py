"""BM25 稀疏检索（自实现 Okapi BM25 + 中文 char-bigram 分词），补 dense 的召回。

不依赖 jieba/rank_bm25：中文用 char bigram（对中文检索足够），英文/数字/LaTeX 按词保留。
"""
import math
import os
import pickle
import re
from collections import Counter

import numpy as np

CJK_RE = re.compile(r'[一-鿿]')
TOKEN_RE = re.compile(r'[a-zA-Z0-9\\_^}{]+|[一-鿿]+')


def tokenize(text):
    tokens = []
    for seg in TOKEN_RE.findall(text):
        if CJK_RE.match(seg):
            if len(seg) == 1:
                tokens.append(seg)
            else:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg.lower())
    return tokens


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.N = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.k1, self.b = k1, b
        df = Counter()
        for d in corpus:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings = {}
        for i, d in enumerate(corpus):
            for t, cnt in Counter(d).items():
                self.postings.setdefault(t, {})[i] = cnt

    def get_scores(self, query_tokens):
        scores = [0.0] * self.N
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf in self.postings.get(t, {}).items():
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


def build_bm25(chunks):
    return BM25([tokenize(c['text']) for c in chunks])


def save_bm25(bm25, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    with open(dir_path / 'bm25.pkl', 'wb') as f:
        pickle.dump(bm25, f)


def load_bm25(dir_path):
    with open(dir_path / 'bm25.pkl', 'rb') as f:
        return pickle.load(f)


def search_bm25(bm25, query, top_k=30):
    scores = np.asarray(bm25.get_scores(tokenize(query)))
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
