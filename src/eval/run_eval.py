"""分层评测运行器：retrieve → generate → judge，按数学基础/统计专业课/章节汇总。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config
from src.ingest.embed import load_model
from src.ingest.index import load_index
from src.rag.retrieve import retrieve
from src.rag.generate import generate
from src.eval.metrics import judge_answer, judge_relevance


def _agg(group):
    if not group:
        return None
    return {
        'n': len(group),
        '答案准确率': round(sum(x['答案分'] for x in group) / len(group), 3),
        '平均相关性': round(sum(x['相关性'] for x in group) / len(group), 2),
    }


def main(limit=None):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    # 主评测：知识点 + 正式 + 非确认型（确认型单列不进主指标）
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式' and not r['确认型']]
    if limit:
        es = es[:limit]

    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)

    rows = []
    for i, r in enumerate(es, 1):
        ret = retrieve(r['问题'], model, index, chunks)
        gen = generate(r['问题'], ret)
        ans = judge_answer(r['问题'], r['答案'], gen)
        rel = judge_relevance(r['问题'], ret)
        rows.append({'id': r['id'], '章节': r['章节'], '答案评分': ans, '相关性': rel,
                     '答案分': {'对': 1, '半对': 0.5, '错': 0}[ans]})
        if i % 10 == 0:
            print(f'[{i}/{len(es)}] 完成', flush=True)

    report = {
        '数学基础': _agg([x for x in rows if x['章节'] == '数学基础']),
        '统计专业课': _agg([x for x in rows if x['章节'] != '数学基础']),
        '按章节': {ch: _agg([x for x in rows if x['章节'] == ch])
                   for ch in sorted({x['章节'] for x in rows})},
    }
    json.dump(report, open(config.DATA_DIR / 'report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
