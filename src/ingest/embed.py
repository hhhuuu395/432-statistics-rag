"""BGE-M3 加载与编码（模型缓存走 config 里的 D 盘 + hf-mirror）。"""
from sentence_transformers import SentenceTransformer

from ..utils import config

_model = None


def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts):
    return load_model().encode(list(texts), normalize_embeddings=True)
