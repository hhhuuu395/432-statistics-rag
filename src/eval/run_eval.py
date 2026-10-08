"""分层评测运行器：retrieve → generate → judge，按数学基础/统计专业课/章节汇总（并发 + 重试）。"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config
from src.ingest.embed import load_model
from src.ingest.index import load_index
from src.rag.retrieve import hybrid_retrieve
from src.rag.bm25 import load_bm25
from src.rag.rerank import load_reranker
from src.rag.generate import generate
from src.eval.metrics import judge_answer, judge_relevance


def _process_one(r, model, index, chunks, bm25, retries=2):
    for attempt in range(retries + 1):
        try:
            ret = hybrid_retrieve(r['问题'], model, index, chunks, bm25)
            gen = generate(r['问题'], ret)
            ans = judge_answer(r['问题'], r['答案'], gen)
            rel = judge_relevance(r['问题'], ret)
            return {'id': r['id'], '章节': r['章节'], '答案评分': ans, '相关性': rel,
                    '答案分': {'对': 1, '半对': 0.5, '错': 0}[ans]}
        except Exception:
            if attempt == retries:
                return {'id': r['id'], '章节': r['章节'], '答案评分': '错', '相关性': 0, '答案分': 0}
            time.sleep(3)


def _agg(group):
    if not group:
        return None
    return {
        'n': len(group),
        '答案准确率': round(sum(x['答案分'] for x in group) / len(group), 3),
        '平均相关性': round(sum(x['相关性'] for x in group) / len(group), 2),
    }


def main(limit=None, workers=8):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式' and not r['确认型']]
    if limit:
        es = es[:limit]

    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)
    bm25 = load_bm25(config.BM25_DIR)
    load_reranker()  # 预加载 reranker，避免多线程竞争

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, r, model, index, chunks, bm25): r for r in es}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 20 == 0:
                print(f'[{done}/{len(es)}] 完成', flush=True)
                # 增量落盘：任务中途挂掉也不丢已完成的部分
                json.dump(rows, open(config.DATA_DIR / 'eval_rows_partial.json', 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=2)

    report = {
        '数学基础': _agg([x for x in rows if x['章节'] == '数学基础']),
        '统计专业课': _agg([x for x in rows if x['章节'] != '数学基础']),
        '按章节': {ch: _agg([x for x in rows if x['章节'] == ch])
                   for ch in sorted({x['章节'] for x in rows})},
    }
    json.dump(report, open(config.DATA_DIR / 'report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(limit, workers)
