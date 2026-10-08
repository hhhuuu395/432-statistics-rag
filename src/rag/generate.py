"""生成：构造 prompt + DeepSeek 生成答案。"""
import requests

from ..utils import config

OLD_PROMPT_TEMPLATE = ("你是中南大学432统计学考研辅导老师。基于下面资料回答学生问题，标注引用来源（如[来源1]）。"
                       "若资料不足以回答，就说资料不足，不要编造。\n\n资料：\n{ctx}\n\n学生问题：{question}\n\n回答：")

NEW_PROMPT_TEMPLATE = ("你是中南大学432统计学考研辅导老师。基于下面资料回答学生问题，标注引用来源（如[来源1]）。\n"
                       "要求：\n"
                       "1. 直接简洁，先给结论再简要解释，不要长篇铺垫或绕圈子。\n"
                       "2. 基于资料尽力回答：即使资料没有逐字相同的答案，也要结合相关概念给出最接近的解答；只有资料与问题完全无关时才说明资料不足。\n"
                       "3. 学生问题若简短或有指代（如「这个题」「这一步」「第一问」），结合资料块推断其具体所指再回答；确实无法判断时才请学生补充。\n\n"
                       "资料：\n{ctx}\n\n学生问题：{question}\n\n回答：")


def _ctx_str(retrieved):
    return '\n\n'.join(
        f"[来源{i + 1} {c['chunk']['source']} p{c['chunk']['page']}]\n{c['chunk']['text']}"
        for i, c in enumerate(retrieved)
    )


def build_prompt(question, retrieved):
    return NEW_PROMPT_TEMPLATE.format(ctx=_ctx_str(retrieved), question=question)


def build_prompt_old(question, retrieved):
    """旧 prompt（用于 A/B 对比「编造」风险）。"""
    return OLD_PROMPT_TEMPLATE.format(ctx=_ctx_str(retrieved), question=question)


def generate(question, retrieved, model=None, prompt_builder=build_prompt):
    model = model or config.LLM_MODEL
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt_builder(question, retrieved)}], 'temperature': 0}
    r = requests.post(config.LLM_API_URL,
                      headers={'Authorization': 'Bearer ' + config.get_key(), 'Content-Type': 'application/json'},
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']
