"""检索：query → top-k 块（带来源与分数）。"""
import numpy as np

from ..utils import config


def retrieve(query, model, index, chunks, top_k=None):
    top_k = top_k or config.TOP_K
    qv = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(qv, dtype='float32'), top_k)
    out = []
    for j, i in enumerate(ids[0]):
        if i == -1:
            continue
        out.append({'chunk': chunks[int(i)], 'score': float(scores[0][j])})
    return out
