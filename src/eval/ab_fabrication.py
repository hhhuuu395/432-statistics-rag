"""A/B 对比：新旧 prompt 的「编造」率（验证改动1 是否矫枉过正）。"""
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import config
from src.ingest.embed import load_model
from src.ingest.index import load_index
from src.rag.bm25 import load_bm25
from src.rag.rerank import load_reranker
from src.rag.retrieve import hybrid_retrieve
from src.rag.generate import generate, build_prompt, build_prompt_old
from src.eval.metrics import judge_fabrication


def _run_one(r, model, index, chunks, bm25, prompt_builder):
    ret = hybrid_retrieve(r['问题'], model, index, chunks, bm25)
    gen = generate(r['问题'], ret, prompt_builder=prompt_builder)
    fab = judge_fabrication(r['问题'], ret, gen)
    return fab


def main(n=30):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式' and not r['确认型']
          and r['章节'] != '数学基础' and r['章节']]
    random.seed(42)
    sample = random.sample(es, n)
    print(f'抽样 {n} 条统计专业课', flush=True)

    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)
    bm25 = load_bm25(config.BM25_DIR)
    load_reranker()

    for name, pb in [('旧prompt', build_prompt_old), ('新prompt', build_prompt)]:
        fabs = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_run_one, r, model, index, chunks, bm25, pb): r for r in sample}
            done = 0
            for fut in as_completed(futs):
                fabs.append(fut.result())
                done += 1
                if done % 10 == 0:
                    print(f'  [{name}] {done}/{n}', flush=True)
        cnt = Counter(fabs)
        print(f'[{name}] 编造分布: {dict(cnt)}  编造率 {cnt["有编造"]}/{n} = {cnt["有编造"]/n:.0%}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
