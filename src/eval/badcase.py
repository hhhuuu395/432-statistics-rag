"""badcase 归因分析：抽样统计专业课样本，逐条归因到 语料缺失/检索未召回/生成没用好/judge误判。"""
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config
from src.ingest.embed import load_model
from src.ingest.index import load_index
from src.rag.retrieve import retrieve
from src.rag.generate import generate
from src.eval.metrics import judge_answer, _ask

CATS = ['语料缺失', '检索未召回', '生成没用好', 'judge误判', '非badcase']


def _classify(问题, gold, ret, gen, ans):
    ctx = '\n'.join(
        f"[{i + 1}] {Path(c['chunk']['source']).name} p{c['chunk']['page']} (score {c['score']:.2f}): {c['chunk']['text'][:150]}"
        for i, c in enumerate(ret)
    )
    prompt = (f"你是RAG系统badcase归因分析器。判断下面这个case的主要问题，归为四类之一。\n\n"
              f"【学生问题】{问题}\n【标准答案】{gold}\n"
              f"【RAG检索到的top-{len(ret)}资料块】\n{ctx}\n"
              f"【RAG生成答案】{gen}\n【judge评分】{ans}\n\n"
              f"归类规则（只输出类别名 + 冒号 + 一句话理由，不要其他文字）：\n"
              f"- 语料缺失：检索块里没有任何能回答该问题的内容，且该问题属于茆诗松教材等未入库语料的知识点"
              f"（当前语料=真题/模拟卷/专题笔记/讲义，无茆书正文）。\n"
              f"- 检索未召回：检索块不相关，但该问题是真题/笔记应覆盖的基础考点，只是没被召回。\n"
              f"- 生成没用好：检索块里有相关内容，但生成答案没正确利用（答错/漏答/答非所问）。\n"
              f"- judge误判：生成答案其实基本正确，judge却判错了。\n"
              f"- 非badcase：这个case答对了，不算失败。")
    resp = _ask(prompt)
    first = resp.strip().splitlines()[0].strip()
    for c in CATS:
        if c in first:
            return c
    return '其他'


def _process_one(r, model, index, chunks):
    ret = retrieve(r['问题'], model, index, chunks, top_k=15)
    gen = generate(r['问题'], ret[:8])
    ans = judge_answer(r['问题'], r['答案'], gen)
    cat = _classify(r['问题'], r['答案'], ret, gen, ans)
    return {
        'id': r['id'], '章节': r['章节'], '问题': r['问题'], '标准答案': r['答案'],
        'judge': ans, '归因': cat, '生成答案': gen[:400],
        '检索块': [{'来源': Path(c['chunk']['source']).name, 'page': c['chunk']['page'],
                    'score': round(c['score'], 2), 'text': c['chunk']['text'][:150]} for c in ret[:5]],
    }


def _stratified_sample(es, n=30):
    by_ch = defaultdict(list)
    for r in es:
        by_ch[r['章节']].append(r)
    sample = []
    for ch, items in by_ch.items():
        k = max(1, round(n * len(items) / len(es)))
        step = max(1, len(items) // k)
        sample.extend(items[::step][:k])
    return sample[:n]


def main(n=30, workers=6):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式' and not r['确认型']
          and r['章节'] != '数学基础' and r['章节'] is not None]
    sample = _stratified_sample(es, n)
    print(f'抽样 {len(sample)} 条（共 {len(es)} 条统计专业课）', flush=True)

    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, r, model, index, chunks): r for r in sample}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 5 == 0:
                print(f'[{done}/{len(sample)}] 完成', flush=True)

    json.dump(rows, open(config.DATA_DIR / 'badcase_analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    cnt = Counter(r['归因'] for r in rows)
    print('\n=== 归因分布 ===', flush=True)
    for c, k in cnt.most_common():
        print(f'{c}: {k} ({k * 100 // len(rows)}%)', flush=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(n, workers)
