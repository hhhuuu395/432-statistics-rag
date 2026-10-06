import sys
sys.path.insert(0, '.')

from src.eval.build_eval_set import clean_zhiku_cell, build_eval_set


def test_clean_zhiku_splits_followup():
    cell = ('院校：中南大学\n 问题：专业课难度怎么样？\n 答案：居中。\n'
            ' 问：考试是简答论述计算吗？\n 对的。\n 6. 通知：')
    recs = clean_zhiku_cell(cell)
    assert len(recs) == 2
    assert recs[0]['问题'] == '专业课难度怎么样？'
    assert recs[1]['问题'] == '考试是简答论述计算吗？'
    assert recs[1]['答案'] == '对的。'


def test_eval_set_has_unique_ids_and_no_empty_question():
    es = build_eval_set()
    ids = [r['id'] for r in es]
    assert len(ids) == len(set(ids))
    assert all(r['问题'].strip() for r in es)
    missing = [r for r in es if r['状态'] == '答案缺失']
    assert len(missing) == 22 and all(r['答案'] is None for r in missing)
