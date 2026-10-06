"""评测指标：LLM-as-judge（答案对错 + 检索相关性）。"""
import re

import requests

from ..utils import config


def _ask(prompt, model=None):
    model = model or config.LLM_MODEL
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0}
    r = requests.post(config.LLM_API_URL,
                      headers={'Authorization': 'Bearer ' + config.get_key(), 'Content-Type': 'application/json'},
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']


def _parse_judge(content):
    if '半对' in content:
        return '半对'
    if '对' in content:
        return '对'
    return '错'


def judge_answer(question, gold, generated, model=None):
    prompt = (f"你是432统计学答疑评测裁判。判断生成答案是否正确。\n问题：{question}\n"
              f"标准答案：{gold}\n生成答案：{generated}\n只输出：对 / 半对 / 错")
    return _parse_judge(_ask(prompt, model))


def judge_relevance(question, chunks, model=None):
    ctx = '\n'.join(f"[{i + 1}] {c['chunk']['text'][:300]}" for i, c in enumerate(chunks))
    prompt = (f"下面是检索到的资料。判断这些资料是否足以回答该问题。\n问题：{question}\n资料：\n{ctx}\n"
              f"只输出一个数字：3=充分，2=部分，1=不足")
    content = _ask(prompt, model)
    m = re.search(r'[123]', content)
    return int(m.group(0)) if m else 0
