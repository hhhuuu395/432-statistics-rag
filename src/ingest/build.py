"""从 materials 解析 → 分块 → 嵌入 → 建 FAISS 索引并落盘。"""
import json

from ..utils import config
from .parse import parse_directory
from .chunk import chunk_text
from .embed import load_model
from .index import build_index, save_index


def _clean_ocr_text(text):
    """清洗 OCR 出的茆书页文本：去水印、去模型前言、去 markdown 标题符号。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not any(w in ln for w in config.WATERMARK_PATTERNS)]
    lines = [ln for ln in lines if not ln.startswith('以下')]
    lines = [ln.lstrip('#').strip() for ln in lines]
    return '\n'.join(lines)


def build_from_materials():
    """仅文本源（无茆书）。"""
    recs = parse_directory(config.MATERIALS_DIR)
    chunks = [c for r in recs for c in chunk_text(r['text'], r['source'], r['page'])]
    idx, ch = build_index(chunks, load_model())
    save_index(idx, ch, config.INDEX_DIR)
    return len(chunks)


def build_with_maoshu():
    """文本源 + 茆书 OCR 页 → 重建索引。page 转成印刷页码（PDF 页 − 22）。"""
    recs = parse_directory(config.MATERIALS_DIR)
    ocr = json.load(open(config.DATA_DIR / 'maoshu_ocr.json', encoding='utf-8'))
    for r in ocr:
        text = _clean_ocr_text(r['text'])
        if text:
            recs.append({'text': text, 'page': r['page'] - 22, 'source': '茆诗松教材第三版.pdf'})
    chunks = [c for r in recs for c in chunk_text(r['text'], r['source'], r['page'])]
    idx, ch = build_index(chunks, load_model())
    save_index(idx, ch, config.INDEX_DIR)
    return len(chunks)
