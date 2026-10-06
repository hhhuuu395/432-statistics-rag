import sys
sys.path.insert(0, '.')

from src.ingest.chunk import chunk_text


def test_chunk_splits_and_overlaps():
    text = '甲' * 1000
    chunks = chunk_text(text, 's', 1, chunk_size=512, overlap=64)
    assert len(chunks) == 3
    assert all(len(c['text']) <= 512 for c in chunks)


def test_short_text_single_chunk():
    chunks = chunk_text('短文本', 's', 1)
    assert len(chunks) == 1
    assert chunks[0]['text'] == '短文本'
