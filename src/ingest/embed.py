"""BGE-M3 加载与编码（模型缓存走 config 里的 D 盘 + hf-mirror）。"""
# 必须先 import config（它设置 HF_ENDPOINT/HF_HOME/NO_PROXY），
# 否则 sentence_transformers→huggingface_hub 在 import 时就直连 huggingface.co
from ..utils import config
from sentence_transformers import SentenceTransformer

_model = None


def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts):
    return load_model().encode(list(texts), normalize_embeddings=True)
