import sys
sys.path.insert(0, '.')

from src.ingest.parse import _clean


def test_clean_removes_watermark_and_blank():
    t = '统计战线出品\n\n正文内容\n微信公众号：就现在考研\n'
    out = _clean(t)
    assert '统计战线' not in out
    assert '微信公众号' not in out
    assert '正文内容' in out
