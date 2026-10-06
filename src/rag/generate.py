"""生成：构造 prompt + DeepSeek 生成答案。"""
import requests

from ..utils import config


def build_prompt(question, retrieved):
    ctx = '\n\n'.join(
        f"[来源{i + 1} {c['chunk']['source']} p{c['chunk']['page']}]\n{c['chunk']['text']}"
        for i, c in enumerate(retrieved)
    )
    return (f"你是中南大学432统计学考研辅导老师。基于下面资料回答学生问题，标注引用来源（如[来源1]）。"
            f"若资料不足以回答，就说资料不足，不要编造。\n\n资料：\n{ctx}\n\n学生问题：{question}\n\n回答：")


def generate(question, retrieved, model=None):
    model = model or config.LLM_MODEL
    payload = {'model': model, 'messages': [{'role': 'user', 'content': build_prompt(question, retrieved)}], 'temperature': 0}
    r = requests.post(config.LLM_API_URL,
                      headers={'Authorization': 'Bearer ' + config.get_key(), 'Content-Type': 'application/json'},
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']
