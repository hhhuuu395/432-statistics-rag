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


def hybrid_retrieve(query, model, index, chunks, bm25, dense_k=None, bm25_k=None, top_k=None):
    """dense + BM25 → RRF 融合 → reranker 重排。"""
    from .bm25 import search_bm25
    from .rerank import rerank
    dense_k = dense_k or config.DENSE_K
    bm25_k = bm25_k or config.BM25_K
    top_k = top_k or config.TOP_K

    qv = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(qv, dtype='float32'), dense_k)
    dense_hits = [(int(ids[0][j]), float(scores[0][j])) for j in range(ids[0].shape[0]) if ids[0][j] != -1]
    bm25_hits = search_bm25(bm25, query, top_k=bm25_k)

    fused = {}
    for rank, (idx, _) in enumerate(dense_hits):
        fused[idx] = fused.get(idx, 0) + 1.0 / (60 + rank + 1)
    for rank, (idx, _) in enumerate(bm25_hits):
        fused[idx] = fused.get(idx, 0) + 1.0 / (60 + rank + 1)
    candidate_idx = sorted(fused, key=lambda i: -fused[i])
    candidates = [chunks[i] for i in candidate_idx]
    return rerank(query, candidates, top_k=top_k)
