# RAG 基线（Step 1）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通 RAG 最小闭环：文本语料 → 索引 → 检索 → DeepSeek 生成 → 分层评测，产出第一份可复现的评测基线。

**Architecture:** 单进程 Python 脚本流水线。`ingest` 把 materials 里的文本源解析成块并建 FAISS 索引；`rag` 检索 + DeepSeek 生成；`eval` 合并评测集并用 LLM-as-judge 分层打分。先 dense 检索出基线，不引入混合检索/重排/OCR。

**Tech Stack:** Python 3、PyMuPDF、python-docx、openpyxl、python-pptx、sentence-transformers(BGE-M3)、faiss-cpu、requests(DeepSeek)、pytest。

**Spec:** [docs/design.md](../design.md)

## Global Constraints

- Python 3，依赖见 Task 1；不引入 Web 框架、数据库、向量服务（Milvus/Chroma）。
- 生成/判断模型统一走 DeepSeek API，key 从 `C:/Users/86130/Desktop/考研ai/.env` 读 `DEEPSEEK_API_KEY`。
- 所有产物写入 `data/`（评测集、索引、报告），不得写入 `materials/`。
- 检索结果必须带 `source`（文件路径）+ `page`，供引用溯源与评测。
- 评测分层：统计专业课为主指标，数学基础单列，确认型/答案缺失不进主指标（见 design.md §5.6）。
- 每完成一个 Task 就 commit，commit message 用 `feat:`/`test:` 前缀。

## Review Focus

1. **空文档/空文本**：解析出空串或纯水印的文档不应产出块（否则污染索引）。→ Task 3 测试。
2. **块跨页/重叠越界**：chunk 在文本尾部不能出现负 start 或死循环。→ Task 4 测试。
3. **检索返回 -1（FAISS 空索引/结果不足 top-k）**：必须过滤 `id == -1`，否则崩溃。→ Task 6 测试。
4. **评测集合并后 id 唯一、无空问题**：重复 id 或空问题会让评测结果对不上号。→ Task 2 测试。
5. **DeepSeek 返回非 JSON / 超时**：judge 输出解析失败要有兜底（记 0 分而不是崩）。→ Task 8 测试。
6. **中文按字符切块**：chunk_size 是「字符」不是「字节」，否则中英文混排长度错乱。→ Task 4 测试。

---

## 文件结构

```
src/
  utils/config.py          # 路径、常量、key 读取
  ingest/parse.py          # pdf/docx/xlsx/pptx → 统一文本记录
  ingest/chunk.py          # 文本 → 块
  ingest/embed.py          # BGE-M3 加载与编码
  ingest/index.py          # FAISS 建索引/存取
  rag/retrieve.py          # query → top-k 块
  rag/generate.py          # prompt 构造 + DeepSeek 生成
  eval/build_eval_set.py   # 智库+截图+待补录 → data/eval_set.json
  eval/metrics.py          # LLM-as-judge 打分
  eval/run_eval.py         # 分层评测 + 报告
tests/
  test_parse.py  test_chunk.py  test_build_eval_set.py  test_retrieve.py  test_metrics.py
data/                       # 产物（eval_set.json、faiss_index/、report.json）
```

---

### Task 1: 项目脚手架 + 依赖 + config

**Files:**
- Create: `src/__init__.py`, `src/utils/__init__.py`, `src/utils/config.py`, `src/ingest/__init__.py`, `src/rag/__init__.py`, `src/eval/__init__.py`, `requirements.txt`

**Interfaces:**
- Produces: `config.ROOT`, `config.DATA_DIR`, `config.MATERIALS_DIR`, `config.INDEX_DIR`, `config.EMBED_MODEL`, `config.EMBED_DIM`, `config.TOP_K`, `config.LLM_MODEL`, `config.LLM_API_URL`, `config.WATERMARK_PATTERNS`, `config.get_key()`

- [ ] **Step 1: 写 requirements.txt 并安装**

```
# requirements.txt
faiss-cpu
sentence-transformers
pytest
```

Run: `pip install -r requirements.txt`

- [ ] **Step 2: 写 config.py**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
MATERIALS_DIR = ROOT / 'materials'
INDEX_DIR = DATA_DIR / 'faiss_index'

EMBED_MODEL = 'BAAI/bge-m3'
EMBED_DIM = 1024
TOP_K = 8
LLM_MODEL = 'deepseek-flash'
LLM_API_URL = 'https://api.deepseek.com/chat/completions'

WATERMARK_PATTERNS = ['统计战线', '就现在考研', '微信公众号', '关注微信', '出品']

def get_key() -> str:
    return (ROOT / '.env').read_text(encoding='utf-8').strip().split('=', 1)[1].strip()
```

- [ ] **Step 3: 建空包 __init__.py**（`src/`、`src/utils/`、`src/ingest/`、`src/rag/`、`src/eval/` 各一个空文件）

- [ ] **Step 4: 验证可导入**

Run: `python -c "import sys; sys.path.insert(0,'.'); from src.utils import config; print(config.ROOT, config.get_key()[:4])"` — 应打印路径与 key 前 4 位，无报错。

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/
git commit -m "chore: 项目脚手架与配置"
```

---

### Task 2: 评测集合并 → data/eval_set.json

**Files:**
- Create: `src/eval/build_eval_set.py`
- Test: `tests/test_build_eval_set.py`

**Interfaces:**
- Consumes: `config.ROOT`, `config.DATA_DIR`
- Produces: `build_eval_set() -> list[dict]`（每条含 `id/问题/答案/来源/分类/章节/确认型/状态/来源文件`），写 `data/eval_set.json`；`clean_zhiku_cell(text) -> list[dict]`（智库单元格清洗）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_build_eval_set.py
import json, sys, os
sys.path.insert(0, '.')
from src.eval.build_eval_set import clean_zhiku_cell, build_eval_set

def test_clean_zhiku_splits_followup():
    cell = '院校：中南大学\n 问题：专业课难度怎么样？\n 答案：居中。\n 问：考试是简答论述计算吗？\n 对的。\n 6. 通知：'
    recs = clean_zhiku_cell(cell)
    assert len(recs) == 2
    assert recs[1]['问题'] == '考试是简答论述计算吗？'
    assert recs[1]['答案'] == '对的。'

def test_eval_set_has_unique_ids_and_no_empty_question():
    es = build_eval_set()
    ids = [r['id'] for r in es]
    assert len(ids) == len(set(ids))
    assert all(r['问题'].strip() for r in es)
    # 待补录 22 条全部 状态=答案缺失、答案 None
    missing = [r for r in es if r['状态'] == '答案缺失']
    assert len(missing) == 22 and all(r['答案'] is None for r in missing)
```

- [ ] **Step 2: 跑测试确认失败** → `pytest tests/test_build_eval_set.py -v` 期望 `ModuleNotFoundError`

- [ ] **Step 3: 实现 build_eval_set.py**

```python
import json, re
from ..utils import config

KEY_RE = re.compile(r'^(?:(\d+)[\.、]\s*)?(院校|问题|答案|问|答|追问|补充|通知)\s*[:：]\s*(.*)$')

def clean_zhiku_cell(text):
    records, meta, cur, last = [], {}, None, None
    def flush():
        nonlocal cur
        if cur and (cur.get('问题') or cur.get('答案')):
            cur.update(meta); records.append(cur)
        cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        m = KEY_RE.match(line)
        if m:
            _, key, rest = m.groups()
            if key == '院校': meta['院校'] = rest.strip(); continue
            if key in ('通知', '补充'): flush(); last = None; continue
            if key in ('问题', '问', '追问'):
                flush(); cur = {'问题': rest.strip(), '答案': ''}
                last = 'followup' if key in ('问', '追问') else 'q'
            else:
                if cur is None: cur = {'问题': '', '答案': rest.strip()}
                else: cur['答案'] = (cur['答案'] + ' ' + rest.strip()).strip()
                last = 'a'
        else:
            if cur is None: continue
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
        if not v: continue
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
    # 待补录 → 答案缺失
    for r in json.load(open(config.DATA_DIR / '待补录清单.json', encoding='utf-8')):
        add(r['问题'], None, '真实', '知识点', None, False, '答案缺失', r['来源文件'])
    # 注意：待复查清单（空问答对，无问题文本）不并入 eval_set，留在 data/待复查清单.json 人工抽看
    json.dump(es, open(config.DATA_DIR / 'eval_set.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return es
```

- [ ] **Step 4: 跑测试确认通过** → `pytest tests/test_build_eval_set.py -v` 期望 PASS

- [ ] **Step 5: Commit** → `git add src/eval/build_eval_set.py tests/ data/eval_set.json && git commit -m "feat: 合并产出评测集 eval_set.json"`

---

### Task 3: 文档解析 parse.py

**Files:**
- Create: `src/ingest/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: `config.WATERMARK_PATTERNS`
- Produces: `parse_file(path: Path) -> list[dict]`（每条 `{text, page, source}`）；`parse_directory(dir: Path) -> list[dict]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_parse.py
import sys
sys.path.insert(0, '.')
from src.ingest.parse import _clean

def test_clean_removes_watermark_and_blank():
    t = '统计战线出品\n\n正文内容\n微信公众号：就现在考研\n'
    out = _clean(t)
    assert '统计战线' not in out and '微信公众号' not in out and '正文内容' in out
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 parse.py**

```python
import fitz
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from pathlib import Path
from ..utils import config

def _clean(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not any(w in ln for w in config.WATERMARK_PATTERNS)]
    return '\n'.join(lines)

def parse_pdf(path):
    out = []
    for i, page in enumerate(fitz.open(str(path))):
        t = _clean(page.get_text())
        if t: out.append({'text': t, 'page': i + 1, 'source': str(path)})
    return out

def parse_docx(path):
    d = Document(str(path))
    paras = [p.text for p in d.paragraphs]
    for tbl in d.tables:
        for row in tbl.rows:
            paras.append(' | '.join(c.text for c in row.cells))
    t = _clean('\n'.join(paras))
    return [{'text': t, 'page': 1, 'source': str(path)}] if t else []

def parse_xlsx(path):
    wb = load_workbook(str(path), data_only=True)
    out = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None]
            if cells: out.append(' | '.join(cells))
    t = _clean('\n'.join(out))
    return [{'text': t, 'page': 1, 'source': str(path)}] if t else []

def parse_pptx(path):
    prs = Presentation(str(path))
    out = []
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.text.strip():
                out.append(shape.text)
    t = _clean('\n'.join(out))
    return [{'text': t, 'page': 1, 'source': str(path)}] if t else []

def parse_file(path):
    path = Path(path)
    f = {'pdf': parse_pdf, 'docx': parse_docx, 'xlsx': parse_xlsx, 'pptx': parse_pptx}.get(path.suffix.lower().lstrip('.'))
    return f(path) if f else []

def parse_directory(dir):
    out = []
    for p in Path(dir).rglob('*'):
        if p.is_file():
            out.extend(parse_file(p))
    return out
```

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** → `git commit -m "feat: 文档解析 pdf/docx/xlsx/pptx"`

---

### Task 4: 分块 chunk.py

**Files:**
- Create: `src/ingest/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Produces: `chunk_text(text, source, page, chunk_size=512, overlap=64) -> list[dict]`（每条 `{text, source, page}`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_chunk.py
import sys
sys.path.insert(0, '.')
from src.ingest.chunk import chunk_text

def test_chunk_splits_and_overlaps():
    text = '甲' * 1000
    chunks = chunk_text(text, 's', 1, chunk_size=512, overlap=64)
    assert len(chunks) == 3  # 1000 → 512 + (512-64) + (512-64)=... 实际3块
    assert all(len(c['text']) <= 512 for c in chunks)

def test_short_text_single_chunk():
    chunks = chunk_text('短文本', 's', 1)
    assert len(chunks) == 1 and chunks[0]['text'] == '短文本'
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 chunk.py**

```python
def chunk_text(text, source, page, chunk_size=512, overlap=64):
    if len(text) <= chunk_size:
        return [{'text': text, 'source': source, 'page': page}]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({'text': text[start:end], 'source': source, 'page': page})
        if end == n:
            break
        start = end - overlap
    return chunks
```

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit**

---

### Task 5: Embedding + 索引（embed.py + index.py）

**Files:**
- Create: `src/ingest/embed.py`, `src/ingest/index.py`

**Interfaces:**
- Consumes: `config.EMBED_MODEL`, `config.EMBED_DIM`, `config.INDEX_DIR`
- Produces: `embed.load_model() -> SentenceTransformer`；`embed.embed(texts) -> np.ndarray`；`index.build_index(chunks, model) -> (faiss.Index, list[dict])`；`index.save_index(index, chunks, dir)`；`index.load_index(dir) -> (index, chunks)`

- [ ] **Step 1: 实现 embed.py**

```python
from sentence_transformers import SentenceTransformer
from ..utils import config
_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model

def embed(texts):
    return load_model().encode(list(texts), normalize_embeddings=True)
```

- [ ] **Step 2: 实现 index.py**

```python
import faiss, numpy as np, pickle, os
from ..utils import config

def build_index(chunks, model):
    vecs = model.encode([c['text'] for c in chunks], normalize_embeddings=True)
    idx = faiss.IndexFlatIP(config.EMBED_DIM)
    idx.add(np.asarray(vecs, dtype='float32'))
    return idx, chunks

def save_index(index, chunks, dir):
    os.makedirs(dir, exist_ok=True)
    faiss.write_index(index, str(dir / 'index.faiss'))
    pickle.dump(chunks, open(dir / 'chunks.pkl', 'wb'))

def load_index(dir):
    idx = faiss.read_index(str(dir / 'index.faiss'))
    chunks = pickle.load(open(dir / 'chunks.pkl', 'rb'))
    return idx, chunks
```

- [ ] **Step 3: 验证**：`python -c "import sys; sys.path.insert(0,'.'); from src.ingest.embed import load_model; m=load_model(); import numpy as np; v=m.encode(['你好'],normalize_embeddings=True); print(v.shape)"` 期望打印 `(1, 1024)`（首次会下载 BGE-M3 权重，约 2GB，需几分钟）。

- [ ] **Step 4: Commit**

---

### Task 6: 检索 retrieve.py

**Files:**
- Create: `src/rag/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `config.TOP_K`
- Produces: `retrieve(query, model, index, chunks, top_k=config.TOP_K) -> list[dict]`（每条 `{chunk, score}`）

- [ ] **Step 1: 写失败测试（用假索引，不依赖真实 embedding）**

```python
# tests/test_retrieve.py
import sys, numpy as np, faiss
sys.path.insert(0, '.')
from src.rag.retrieve import retrieve

class FakeModel:
    def encode(self, texts, normalize_embeddings=False):
        # 按文本长度给一个可预测向量，query 与 chunk 相同长度则内积最大
        return np.array([[1.0, 0.0] if len(t) % 2 == 0 else [0.0, 1.0] for t in texts], dtype='float32')

def test_retrieve_filters_minus_one():
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([[1.0, 0.0]], dtype='float32'))  # 只有 1 个块
    chunks = [{'text': 'ab', 'source': 's', 'page': 1}]
    out = retrieve('ab', FakeModel(), idx, chunks, top_k=5)
    assert len(out) == 1 and out[0]['chunk']['text'] == 'ab'
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 retrieve.py**

```python
import numpy as np
from ..utils import config

def retrieve(query, model, index, chunks, top_k=None):
    top_k = top_k or config.TOP_K
    qv = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(qv, dtype='float32'), top_k)
    out = []
    for j, i in enumerate(ids[0]):
        if i == -1:
            continue
        out.append({'chunk': chunks[int(i)], 'score': float(scores[0][j])})
    return out
```

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit**

---

### Task 7: 生成 generate.py

**Files:**
- Create: `src/rag/generate.py`

**Interfaces:**
- Consumes: `config.LLM_API_URL`, `config.LLM_MODEL`, `config.get_key()`
- Produces: `build_prompt(question, retrieved) -> str`；`generate(question, retrieved, model=None) -> str`

- [ ] **Step 1: 实现 build_prompt + generate**

```python
import requests
from ..utils import config

def build_prompt(question, retrieved):
    ctx = '\n\n'.join(
        f"[来源{i+1} {c['chunk']['source']} p{c['chunk']['page']}]\n{c['chunk']['text']}"
        for i, c in enumerate(retrieved)
    )
    return (f"你是中南大学432统计学考研辅导老师。基于下面资料回答学生问题，标注引用来源（如[来源1]）。"
            f"若资料不足以回答，就说资料不足，不要编造。\n\n资料：\n{ctx}\n\n学生问题：{question}\n\n回答：")

def generate(question, retrieved, model=None):
    model = model or config.LLM_MODEL
    payload = {'model': model, 'messages': [{'role': 'user', 'content': build_prompt(question, retrieved)}], 'temperature': 0}
    r = requests.post(config.LLM_API_URL, headers={'Authorization': 'Bearer ' + config.get_key(),
                      'Content-Type': 'application/json'}, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']
```

- [ ] **Step 2: 冒烟验证**：`python -c "import sys; sys.path.insert(0,'.'); from src.rag.generate import generate; print(generate('什么是置信区间', [{'chunk':{'text':'置信区间是...','source':'s','page':1}}]))"` 期望返回一段中文回答。

- [ ] **Step 3: Commit**

---

### Task 8: 评测指标 metrics.py

**Files:**
- Create: `src/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `config.get_key()`
- Produces: `judge_answer(question, gold, generated, model=None) -> str`（`对/半对/错`）；`judge_relevance(question, chunks, model=None) -> int`（1–3）；`_parse_judge(content) -> str`（解析失败返回 `'错'` 或 `0` 兜底）

- [ ] **Step 1: 写失败测试（测解析兜底）**

```python
# tests/test_metrics.py
import sys
sys.path.insert(0, '.')
from src.eval.metrics import _parse_judge

def test_parse_judge_fallback_on_garbage():
    assert _parse_judge('这是随便一段话，没有评分') == '错'

def test_parse_judge_extracts_label():
    assert _parse_judge('结论：对') == '对'
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 metrics.py**

```python
import json, re, requests
from ..utils import config

def _ask(prompt, model=None):
    model = model or config.LLM_MODEL
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0}
    r = requests.post(config.LLM_API_URL, headers={'Authorization': 'Bearer ' + config.get_key(),
                      'Content-Type': 'application/json'}, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def _parse_judge(content):
    if '半对' in content: return '半对'
    if '对' in content: return '对'
    return '错'

def judge_answer(question, gold, generated, model=None):
    prompt = (f"你是432统计学答疑评测裁判。判断生成答案是否正确。\n问题：{question}\n"
              f"标准答案：{gold}\n生成答案：{generated}\n只输出：对 / 半对 / 错")
    return _parse_judge(_ask(prompt, model))

def judge_relevance(question, chunks, model=None):
    ctx = '\n'.join(f"[{i+1}] {c['chunk']['text'][:300]}" for i, c in enumerate(chunks))
    prompt = (f"下面是检索到的资料。判断这些资料是否足以回答该问题。\n问题：{question}\n资料：\n{ctx}\n"
              f"只输出一个数字：3=充分，2=部分，1=不足")
    content = _ask(prompt, model)
    m = re.search(r'[123]', content)
    return int(m.group(0)) if m else 0
```

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit**

---

### Task 9: 评测运行器 run_eval.py（分层报告）

**Files:**
- Create: `src/eval/run_eval.py`

**Interfaces:**
- Consumes: 上述所有模块 + `data/eval_set.json`
- Produces: `data/report.json`（分层统计：数学基础 / 统计专业课 / 各章节的平均 judge 分数与 relevance）

- [ ] **Step 1: 实现 run_eval.py**

```python
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import config
from src.ingest.embed import load_model
from src.ingest.index import load_index
from src.rag.retrieve import retrieve
from src.rag.generate import generate
from src.eval.metrics import judge_answer, judge_relevance

def main(limit=None):
    es = json.load(open(config.DATA_DIR / 'eval_set.json', encoding='utf-8'))
    # 只评「知识点 + 正式」
    es = [r for r in es if r['分类'] == '知识点' and r['状态'] == '正式']
    if limit: es = es[:limit]
    model = load_model()
    index, chunks = load_index(config.INDEX_DIR)
    rows = []
    for r in es:
        ret = retrieve(r['问题'], model, index, chunks)
        gen = generate(r['问题'], ret)
        ans = judge_answer(r['问题'], r['答案'], gen)
        rel = judge_relevance(r['问题'], ret)
        rows.append({'id': r['id'], '章节': r['章节'], '答案评分': ans, '相关性': rel,
                     '答案分': {'对': 1, '半对': 0.5, '错': 0}[ans]})
    # 分层汇总
    from collections import defaultdict
    def agg(group):
        if not group: return None
        return {'n': len(group), '答案准确率': sum(x['答案分'] for x in group) / len(group),
                '平均相关性': sum(x['相关性'] for x in group) / len(group)}
    report = {
        '数学基础': agg([x for x in rows if x['章节'] == '数学基础']),
        '统计专业课': agg([x for x in rows if x['章节'] != '数学基础']),
        '按章节': {ch: agg([x for x in rows if x['章节'] == ch])
                   for ch in sorted({x['章节'] for x in rows})},
    }
    json.dump(report, open(config.DATA_DIR / 'report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
```

- [ ] **Step 2: 先建索引（一次性）**

Run: `python -c "import sys; sys.path.insert(0,'.'); from src.ingest.parse import parse_directory; from src.ingest.chunk import chunk_text; from src.ingest.embed import load_model; from src.ingest.index import build_index, save_index; from src.utils import config; recs=parse_directory(config.MATERIALS_DIR); chunks=[c for r in recs for c in chunk_text(r['text'], r['source'], r['page'])]; print('chunks', len(chunks)); idx,ch=build_index(chunks, load_model()); save_index(idx,ch,config.INDEX_DIR); print('index saved')"`

- [ ] **Step 3: 小样本验证** → `python src/eval/run_eval.py 5` 期望打印 5 条的分层报告，无异常。

- [ ] **Step 4: Commit**

```bash
git add src/ data/report.json
git commit -m "feat: RAG 基线评测运行器与分层报告"
```

---

## 执行后收尾

- 全量跑 `python src/eval/run_eval.py`（127 条统计专业课 + 152 数学基础，约 300 次生成 + 600 次 judge 调用）。
- 把基线结果 + badcase 写入 `docs/experiments.md`，进入 design.md §7 的 Step 2。
