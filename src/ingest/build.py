"""从 materials 解析 → 分块 → 嵌入 → 建 FAISS 索引并落盘。"""
from ..utils import config
from .parse import parse_directory
from .chunk import chunk_text
from .embed import load_model
from .index import build_index, save_index


def build_from_materials():
    recs = parse_directory(config.MATERIALS_DIR)
    chunks = [c for r in recs for c in chunk_text(r['text'], r['source'], r['page'])]
    idx, ch = build_index(chunks, load_model())
    save_index(idx, ch, config.INDEX_DIR)
    return len(chunks)
