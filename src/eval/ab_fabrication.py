"""A/B 对比：新旧 prompt 的「编造」率（验证改动1 是否矫枉过正）。"""
import json
import random
import sys
import time
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


def _run_one(r, model, index, chunks, bm25, prompt_builder, retries=2):
    for attempt in range(retries + 1):
        try:
            ret = hybrid_retrieve(r['问题'], model, index, chunks, bm25)
            gen = generate(r['问题'], ret, prompt_builder=prompt_builder)
            return judge_fabrication(r['问题'], ret, gen)
        except Exception:
            if attempt == retries:
                return '失败'
            time.sleep(3)


def main(n=30, layer='统计'):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式' and not r['确认型'] and r['章节']]
    es = [r for r in es if (r['章节'] == '数学基础') == (layer == '数学')]
    random.seed(42)
    sample = random.sample(es, min(n, len(es)))
    print(f'抽样 {len(sample)}/{len(es)} 条（{layer}层）', flush=True)

    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)
    bm25 = load_bm25(config.BM25_DIR)
    load_reranker()

    # 分层结果分文件落盘，避免覆盖（统计层 -> ab_fabrication.json，数学层 -> ab_fabrication_math.json）
    out_path = config.DATA_DIR / ('ab_fabrication_math.json' if layer == '数学' else 'ab_fabrication.json')
    detail, summary = {}, {}
    for name, pb in [('旧prompt', build_prompt_old), ('新prompt', build_prompt)]:
        fabs = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_run_one, r, model, index, chunks, bm25, pb): r for r in sample}
            done = 0
            for fut in as_completed(futs):
                r = futs[fut]
                fabs[r['id']] = fut.result()
                done += 1
                if done % 10 == 0:
                    print(f'  [{name}] {done}/{n}', flush=True)
        valid = {k: v for k, v in fabs.items() if v != '失败'}  # 失败样本不计入编造率
        cnt = Counter(valid.values())
        summary[name] = {'n': len(valid), '有编造': cnt['有编造'], '失败': len(fabs) - len(valid),
                         '编造率': round(cnt['有编造'] / max(1, len(valid)), 3)}
        print(f'[{name}] 编造分布: {dict(cnt)}  编造率 {cnt["有编造"]}/{len(valid)} = '
              f'{cnt["有编造"]/max(1, len(valid)):.0%}  失败 {len(fabs) - len(valid)}', flush=True)
        # 每完成一个 prompt 版本就落盘，任务中途挂掉也不丢已有结果
        for r in sample:
            row = detail.setdefault(r['id'], {'id': r['id'], '章节': r['章节'], '问题': r['问题'],
                                              '标准答案': r['答案']})
            row[name] = fabs.get(r['id'], '')
        json.dump({'summary': summary, '明细': list(detail.values())},
                  open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f'  -> 已写入 {out_path.name}', flush=True)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    layer = sys.argv[2] if len(sys.argv) > 2 else '统计'
    main(n, layer)
