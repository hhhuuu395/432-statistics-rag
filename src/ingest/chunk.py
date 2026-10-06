"""分块：文本按字符切分，带重叠，保留来源与页码。"""


def chunk_text(text, source, page, chunk_size=512, overlap=64):
    if len(text) <= chunk_size:
        return [{'text': text, 'source': source, 'page': page}]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({'text': text[start:end], 'source': source, 'page': page})
        if end == n:
            break
        start = end - overlap
    return chunks
