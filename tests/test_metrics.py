import sys
sys.path.insert(0, '.')

from src.eval.metrics import _parse_judge


def test_parse_judge_fallback_on_garbage():
    assert _parse_judge('这是随便一段话，没有评分') == '错'


def test_parse_judge_extracts_label():
    assert _parse_judge('结论：对') == '对'
