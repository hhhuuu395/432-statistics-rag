"""BGE-reranker 重排（CrossEncoder 对 query-chunk 对打分）。"""
from sentence_transformers import CrossEncoder

from ..utils import config

_model = None


def load_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(query, candidates, top_k=8):
    """candidates: list of chunk dicts；返回按分数降序的 top_k 个 [{chunk, score}]。"""
    model = load_reranker()
    pairs = [(query, c['text']) for c in candidates]
    scores = model.predict(pairs)
    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    return [{'chunk': candidates[i], 'score': float(scores[i])} for i in order[:top_k]]
