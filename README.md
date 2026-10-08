# 432 统计学 AI 答疑助手

面向中南大学应用统计（432 统计学）考研学生的**知识点答疑助手**：学生提问 → 混合检索教材 / 真题语料 → DeepSeek 生成**带引用来源**的答案。

项目重点不在「把 RAG 搭起来」，而在**用真实答疑记录做评测、由 badcase 数据驱动每一轮改动**。四级迭代把统计专业课答案准确率从 **16.7% 提到 35.7%**，每一步改什么、留不留都由评测数据决定 —— 包括一次「加了复杂组件但评测没涨」的诚实回退判断。

> 完整曲线与归因见 **[docs/final_report.md](docs/final_report.md)**；逐阶段实验记录见 [docs/experiments.md](docs/experiments.md)。

## 系统架构

```
materials/  教材笔记 · 历年真题 · 模拟卷 · 答疑截图
   │  parse    PDF / docx / xlsx / pptx；纯扫描页走多模态 LLM OCR
   ▼
纯文本 + 元数据（source / page）
   │  chunk    ~512 字 + 64 字重叠，保留页码供引用溯源
   ▼
文本块
   ├── embed  BGE-M3（1024 维）──> FAISS dense 召回 top-20
   └── BM25   （自实现，char-bigram）──> 稀疏召回 top-20
   │  RRF 融合（k=60）──> BGE-reranker-v2-m3 重排 ──> top-8
   ▼
DeepSeek 生成 ──> 带 [来源N] 引用的答案

评测闭环：  评测集 ──> 检索 ──> 生成 ──> LLM-as-judge ──> 分层报告
                                      （答案 对/半对/错 · 相关性 1–3）
```

## 技术栈

| 环节 | 选型 |
| --- | --- |
| 生成 / 裁判 / 视觉 OCR | DeepSeek `deepseek-flash`（视觉能力同时用于答疑截图提取和扫描教材 OCR） |
| 向量模型 | BGE-M3（本地部署，1024 维） |
| 重排模型 | BGE-reranker-v2-m3（本地 CrossEncoder） |
| 向量索引 | FAISS（`IndexFlatIP`） |
| 稀疏检索 | 自实现 BM25（char-bigram 分词，k1=1.5 / b=0.75） |
| 混合融合 | RRF（Reciprocal Rank Fusion，k=60） |
| 评测 | LLM-as-judge（答案对/半对/错；检索相关性 1–3） |
| 语言 | Python 3.10 |

## 如何运行

### 0. 环境

```bash
python -m venv .venv
.venv/Scripts/activate                       # Windows；macOS / Linux: source .venv/bin/activate
python -m pip install -U pip                 # 先升级 pip —— 部分国内镜像需要新版 pip 才能解析
pip install -r requirements.txt
```

若默认源慢或不可达，加国内镜像（实测阿里云可解析 `faiss-cpu`，清华源在旧 pip 下会报 `No matching distribution`）：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

> `requirements.txt` **刻意保持纯 ASCII**：pip 在中文 Windows 上按系统本地编码（GBK）解码 requirements 文件，中文注释会直接抛 `UnicodeDecodeError`。

在项目根目录建 `.env`，写入一行（已 gitignore，不进仓库）：

```
DEEPSEEK_API_KEY=sk-...
```

### 1. 下载向量模型（BGE-M3，约 2GB）

```bash
python scripts/download_bge_m3.py     # 从 ModelScope 下载到 D:\86130\models\bge-m3
```

### 2. OCR 扫描版教材（茆诗松《概率论与数理统计》前 6 章）

```bash
python scripts/ocr_maoshu.py 23 335 6     # 参数：PDF 起页 止页 并发数；支持断点续跑
```

### 3. 建索引（FAISS dense + BM25 sparse）

```bash
python -c "from src.ingest.build import build_with_maoshu as b; print(b(), 'chunks')"
```

### 4. 跑评测

```bash
python src/eval/run_eval.py               # 分层评测 → data/report.json
python src/eval/ab_fabrication.py 30 统计  # 编造率 A/B（旧/新 prompt 对比）
```

> `build.py` / `build_eval_set.py` 目前是库函数、没有 CLI 入口，README 里用 `python -c` 调用。

## 评测集说明

| 口径 | 条数 |
| --- | --- |
| `data/eval_set.json` 总条数 | 406 |
| 知识点类 / 规划类 | 401 / 5 |
| 状态：正式 / 答案缺失 | 384 / 22 |
| 确认型 | 97 |
| **主指标集** = 知识点 ∧ 正式 ∧ 非确认型 | **282** |
| ├ 统计专业课 | 129（RAG 主战场） |
| └ 数学基础（数三） | 153（单列，检索帮不上） |

**数据来源**：老师与学生 / 助教的**真实微信答疑记录截图**（417 张 → 提取 377 条知识点问答对）+ 智库 xlsx 整理 7 条。主指标集样本目前**全部来源=真实**，构造题（用于留存验证）尚未产出。

**分层原则（关键）**：数学基础与统计专业课**分开评测**。统计教材对纯数三数学题没有帮助，混评会互相稀释 —— 基线时数学基础只有 4.9%，不是 RAG 的锅。

**两类不进主指标**：
- **确认型** 97 条：答案只是「对的 / 没问题」，没有信息量，进主指标会虚高准确率。
- **答案缺失** 22 条：原题是语音回复、无法提取答案。**问题保留、不删除**，用于真实问题分布统计。

### PII 处理

- 学生信息在**提取时当场脱敏**：姓名 / 昵称 → `[学生N]`，助教 / 学长 → `[学长]`，手机号 / 微信号 → `[脱敏]`，头像直接丢弃。老师本人发言**不脱敏**（属教学内容）。
- `materials/`（原始截图 + 教材 PDF）**不入库**：涉及学生隐私与教材版权，已 gitignore。
- 入库前做 PII 正则扫描（手机号 / 身份证 / 邮箱），当前 `data/` 无命中。

## 目录结构

```
src/
  ingest/   parse.py chunk.py embed.py index.py build.py     # 文档 → 索引
  rag/      retrieve.py bm25.py rerank.py generate.py        # 检索 + 生成
  eval/     metrics.py run_eval.py badcase.py badcase_gen.py
            ab_fabrication.py build_eval_set.py              # 评测 + badcase 归因
  utils/    config.py                                        # 路径 / key / 常量
scripts/    download_bge_m3.py ocr_maoshu.py                 # 模型下载 / 教材 OCR
tests/      parse / chunk / retrieve / metrics 单元测试
data/       评测集 · 索引与报告产物（无 PII，可入库）
docs/       design.md experiments.md final_report.md
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/final_report.md](docs/final_report.md) | **收官报告**：四级优化曲线、每级归因、已知局限、future work |
| [docs/experiments.md](docs/experiments.md) | 逐阶段实验记录与决策理由 |
| [docs/design.md](docs/design.md) | MVP 设计文档与已定决策 |
| [项目说明.md](项目说明.md) | 项目背景、原则、进度 |
