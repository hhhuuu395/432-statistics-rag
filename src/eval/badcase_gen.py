"""第二轮 badcase 分析：聚焦「生成失败」（检索相关性不错但答案错），
归因到 prompt没压住 / 该换更强模型 / judge误判。"""
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config
from src.ingest.embed import load_model
from src.ingest.index import load_index
from src.rag.bm25 import load_bm25
from src.rag.rerank import load_reranker
from src.rag.retrieve import hybrid_retrieve
from src.rag.generate import generate
from src.eval.metrics import judge_answer, judge_relevance, _ask

CATS = ['prompt没压住', '该换更强模型', 'judge误判', '其他']


def _classify(问题, gold, ret, gen, ans):
    ctx = '\n'.join(f"[{i + 1}] {Path(c['chunk']['source']).name} p{c['chunk']['page']}: {c['chunk']['text'][:120]}"
                    for i, c in enumerate(ret))
    prompt = (f"你是RAG badcase归因分析器（第二轮，聚焦生成失败）。这个case检索到了相关内容，但生成答案仍被judge判错。"
              f"判断失败根因，归为三类之一。\n\n"
              f"【学生问题】{问题}\n【标准答案】{gold}\n【检索到的资料块】\n{ctx}\n【生成答案】{gen}\n【judge评分】{ans}\n\n"
              f"归类（只输出类别名 + 冒号 + 一句话理由）：\n"
              f"- prompt没压住：资料里其实有答案，但生成没正确用上（答非所问/漏要点/误判资料不足/没按指令引用）。\n"
              f"- 该换更强模型：题目本身需要较强推理/计算/证明（deepseek-flash 能力不够），换 deepseek-v4-pro 可能才对。\n"
              f"- judge误判：生成答案其实基本正确，judge 却判错了。")
    resp = _ask(prompt)
    first = resp.strip().splitlines()[0].strip()
    for c in CATS:
        if c in first:
            return c
    return '其他'


def _process_one(r, model, index, chunks, bm25):
    try:
        ret = hybrid_retrieve(r['问题'], model, index, chunks, bm25)
        gen = generate(r['问题'], ret)
        ans = judge_answer(r['问题'], r['答案'], gen)
        rel = judge_relevance(r['问题'], ret)
        cat = ''
        if ans != '对' and rel >= 2:  # 生成失败：检索到了但答案错
            cat = _classify(r['问题'], r['答案'], ret, gen, ans)
        return {'id': r['id'], '章节': r['章节'], '问题': r['问题'], '标准答案': r['答案'],
                'judge': ans, '相关性': rel, '归因': cat, '生成答案': gen[:300],
                '检索块': [{'来源': Path(c['chunk']['source']).name, 'page': c['chunk']['page'],
                            'text': c['chunk']['text'][:100]} for c in ret[:3]]}
    except Exception as e:
        return {'id': r['id'], '章节': r['章节'], '问题': r['问题'], 'judge': '错', '相关性': 0,
                '归因': '', '生成答案': str(e)[:100], '检索块': []}


def _stratified_sample(es, n):
    by_ch = defaultdict(list)
    for r in es:
        by_ch[r['章节']].append(r)
    sample = []
    for ch, items in by_ch.items():
        k = max(1, round(n * len(items) / len(es)))
        step = max(1, len(items) // k)
        sample.extend(items[::step][:k])
    return sample[:n]


def main(n=35, workers=6):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式' and not r['确认型']
          and r['章节'] != '数学基础' and r['章节'] is not None]
    sample = _stratified_sample(es, n)
    print(f'抽样 {len(sample)} 条（共 {len(es)} 条统计专业课）', flush=True)

    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)
    bm25 = load_bm25(config.BM25_DIR)
    load_reranker()

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, r, model, index, chunks, bm25): r for r in sample}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 5 == 0:
                print(f'[{done}/{len(sample)}] 完成', flush=True)

    json.dump(rows, open(config.DATA_DIR / 'badcase_gen.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    gen_fail = [r for r in rows if r['归因']]
    cnt = Counter(r['归因'] for r in gen_fail)
    print(f'\n=== 生成失败 {len(gen_fail)} 条，归因分布 ===', flush=True)
    for c, k in cnt.most_common():
        print(f'{c}: {k} ({k * 100 // max(1, len(gen_fail))}%)', flush=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(n, workers)
