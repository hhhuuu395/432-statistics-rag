import sys

import faiss
import numpy as np

sys.path.insert(0, '.')

from src.rag.retrieve import retrieve


class FakeModel:
    def encode(self, texts, normalize_embeddings=False):
        return np.array([[1.0, 0.0] if len(t) % 2 == 0 else [0.0, 1.0] for t in texts], dtype='float32')


def test_retrieve_filters_minus_one():
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([[1.0, 0.0]], dtype='float32'))
    chunks = [{'text': 'ab', 'source': 's', 'page': 1}]
    out = retrieve('ab', FakeModel(), idx, chunks, top_k=5)
    assert len(out) == 1
    assert out[0]['chunk']['text'] == 'ab'
