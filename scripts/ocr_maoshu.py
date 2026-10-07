"""茆书扫描页 OCR：deepseek-flash 视觉逐页转写，输出 {page, chapter, text}。

用法：python scripts/ocr_maoshu.py [start_pdf_page] [end_pdf_page] [workers]
默认 OCR 前 6 章（PDF 23–335）。断点续跑：已存在于输出 JSON 的页会跳过。
"""
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')

import fitz
import requests

from src.utils import config

PDF = r'materials/教材笔记/茆诗松教材第三版.pdf'
OUT = config.DATA_DIR / 'maoshu_ocr.json'
DPI = 150

# 前 6 章：PDF 起始页、结束页、章节名
CHAPTERS = [
    (23, 76, '随机事件与概率'),
    (77, 146, '随机变量及其分布'),
    (147, 207, '多维随机变量及其分布'),
    (208, 244, '大数定律与中心极限定理'),
    (245, 288, '统计量及其分布'),
    (289, 335, '参数估计'),
]

PROMPT = ("这是茆诗松《概率论与数理统计教程》（第三版）的一页正文。请完整转写本页内容："
          "保留章节标题、定义/定理/例题/性质的编号与文字；数学公式用 LaTeX 表示；"
          "图表若不影响理解可略过但保留文字说明。逐段完整转写，不要总结、不要省略、不要添加解释。")


def chapter_of(page):
    for s, e, ch in CHAPTERS:
        if s <= page <= e:
            return ch
    return '未知'


def ocr_page(page, doc):
    pix = doc[page - 1].get_pixmap(dpi=DPI)
    b64 = base64.b64encode(pix.tobytes('png')).decode()
    payload = {'model': config.LLM_MODEL, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': PROMPT},
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
    ]}], 'temperature': 0}
    r = requests.post(config.LLM_API_URL,
                      headers={'Authorization': 'Bearer ' + config.get_key(), 'Content-Type': 'application/json'},
                      json=payload, timeout=180)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def main(start=23, end=335, workers=6):
    doc = fitz.open(PDF)
    # 断点续跑
    done = {}
    if os.path.exists(OUT):
        done = {r['page']: r for r in json.load(open(OUT, encoding='utf-8'))}
    pages = [p for p in range(start, end + 1) if p not in done]
    print(f'待 OCR {len(pages)} 页（已跳过 {len(done)} 页）', flush=True)

    def run(p):
        for _ in range(3):
            try:
                return {'page': p, 'chapter': chapter_of(p), 'text': ocr_page(p, doc)}
            except Exception:
                import time
                time.sleep(2)
        return {'page': p, 'chapter': chapter_of(p), 'text': ''}

    results = dict(done)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, p): p for p in pages}
        n = 0
        for fut in as_completed(futs):
            r = fut.result()
            results[r['page']] = r
            n += 1
            if n % 10 == 0:
                # 增量落盘
                json.dump([results[k] for k in sorted(results)], open(OUT, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
                print(f'[{n}/{len(pages)}] 完成，已落盘', flush=True)
    json.dump([results[k] for k in sorted(results)], open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'全部完成，共 {len(results)} 页 -> {OUT}', flush=True)


if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 335
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    main(start, end, workers)
