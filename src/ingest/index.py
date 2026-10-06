"""FAISS 索引：建库、存取。"""
import os
import pickle

import faiss
import numpy as np

from ..utils import config


def build_index(chunks, model):
    vecs = model.encode([c['text'] for c in chunks], normalize_embeddings=True)
    idx = faiss.IndexFlatIP(config.EMBED_DIM)
    idx.add(np.asarray(vecs, dtype='float32'))
    return idx, chunks


def save_index(index, chunks, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    faiss.write_index(index, str(dir_path / 'index.faiss'))
    with open(dir_path / 'chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)


def load_index(dir_path):
    idx = faiss.read_index(str(dir_path / 'index.faiss'))
    with open(dir_path / 'chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    return idx, chunks
