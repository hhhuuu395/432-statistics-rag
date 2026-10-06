"""合并智库 + 截图问答对 + 待补录 → data/eval_set.json（统一评测集 schema）。"""
import json
import re

from ..utils import config

KEY_RE = re.compile(r'^(?:(\d+)[\.、]\s*)?(院校|问题|答案|问|答|追问|补充|通知)\s*[:：]\s*(.*)$')


def clean_zhiku_cell(text):
    """把一个智库单元格（多行键值）解析成若干条 {问题, 答案, 院校}。"""
    records, meta, cur, last = [], {}, None, None

    def flush():
        nonlocal cur
        if cur and (cur.get('问题') or cur.get('答案')):
            cur.update(meta)
            records.append(cur)
        cur = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = KEY_RE.match(line)
        if m:
            _, key, rest = m.groups()
            if key == '院校':
                meta['院校'] = rest.strip()
                continue
            if key in ('通知', '补充'):
                flush()
                last = None
                continue
            if key in ('问题', '问', '追问'):
                flush()
                cur = {'问题': rest.strip(), '答案': ''}
                last = 'followup' if key in ('问', '追问') else 'q'
            else:
                if cur is None:
                    cur = {'问题': '', '答案': rest.strip()}
                else:
                    cur['答案'] = (cur['答案'] + ' ' + rest.strip()).strip()
                last = 'a'
        else:
            if cur is None:
                continue
            if last in ('a', 'followup') or cur['答案']:
                cur['答案'] = (cur['答案'] + ' ' + line).strip()
            else:
                cur['问题'] = (cur['问题'] + ' ' + line).strip()
    flush()
    return records


# 智库 7 条的人工分类（知识点/规划）+ 章节（按问题前缀匹配，避免全文逐字比对出错）
ZHIKU_META = [
    ('想问问现阶段专业课是学贾书', '规划', None),
    ('想问问中南专业课参数估计会考最大似然估计', '知识点', '参数估计'),
    ('学长，参数方程数三不是不用掌握', '知识点', '数学基础'),
    ('中南的专业课难度怎么样', '规划', None),
    ('考试是简答论述计算吗', '规划', None),
    ('中南真题类似院校推荐', '规划', None),
    ('专业课第一次复习感觉最大的问题是记忆', '规划', None),
]


def _zhiku_meta(问题):
    for prefix, cat, ch in ZHIKU_META:
        if 问题.startswith(prefix):
            return cat, ch
    return '规划', None


def _load_zhiku():
    import openpyxl
    wb = openpyxl.load_workbook(config.MATERIALS_DIR / '答疑记录' / '中南大学智库.xlsx', data_only=True)
    out = []
    for row in wb.active.iter_rows(values_only=True):
        v = row[0]
        if not v:
            continue
        out.extend(clean_zhiku_cell(str(v)))
    return out


def build_eval_set():
    es, n = [], 0

    def add(问题, 答案, 来源, 分类, 章节, 确认型, 状态, 来源文件):
        nonlocal n
        n += 1
        es.append({'id': f'e{n:04d}', '问题': 问题, '答案': 答案, '来源': 来源, '分类': 分类,
                   '章节': 章节, '确认型': 确认型, '状态': 状态, '来源文件': 来源文件})

    # 智库
    for r in _load_zhiku():
        cat, ch = _zhiku_meta(r['问题'])
        add(r['问题'], r['答案'], '真实', cat, ch, False, '正式', '智库.xlsx')

    # 截图问答对
    qa = json.load(open(config.DATA_DIR / 'extracted_qa.json', encoding='utf-8'))
    ch_map = {r['问题']: r['章节'] for r in json.load(open(config.DATA_DIR / 'chapter_distribution.json', encoding='utf-8'))}
    for r in qa:
        add(r['问题'], r['答案'], '真实', '知识点', ch_map.get(r['问题'], '其他'), r['确认型'], '正式', r['来源文件'])

    # 待补录 → 答案缺失（不进评测，问题保留用于真实问题分布统计）
    for r in json.load(open(config.DATA_DIR / '待补录清单.json', encoding='utf-8')):
        add(r['问题'], None, '真实', '知识点', None, False, '答案缺失', r['来源文件'])

    # 注意：待复查清单（空问答对、无问题文本）不并入 eval_set，留在 data/待复查清单.json 人工抽看
    json.dump(es, open(config.DATA_DIR / 'eval_set.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return es
