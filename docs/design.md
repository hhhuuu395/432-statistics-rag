# 432 统计学 AI 答疑助手 — MVP 设计文档

> 状态：待评审。本文档固化方案 C（评测驱动渐进式 RAG）及本阶段所有决策。
> 配套：docs/experiments.md（实验记录）、项目说明.md（进度）。

## 1. 目标与非目标

**目标**：给中南 432 统计考研学生一个「知识点答疑」助手，回答有依据、可溯源；用真实答疑记录做评测，展示「评测驱动迭代」。

**非目标（MVP 不做）**：Web 界面、账号/部署、多轮会话记忆、规划类问题深度支持、全量教材 OCR。

## 2. 已定决策（本阶段）

| 决策 | 结论 |
|---|---|
| RAG 方案 | **C：评测驱动渐进式**（先朴素 RAG 出基线 → badcase 分析 → 按需 OCR 茆书/加混合检索） |
| 生成模型 | DeepSeek（`deepseek-flash` 起步，复杂题切 `deepseek-v4-pro`） |
| 截图提取模型 | DeepSeek `deepseek-flash`（视觉），兜底 Qwen-VL |
| 22 条语音待补录 | **全部不补**，标「答案缺失，不进评测」，问题保留用于真实问题分布统计 |
| 贾书《统计学》扫描版 | **暂不 OCR**，等 badcase 分析显示需要再处理 |
| 脱敏 | 提取时当场改写（学生→[学生N]、助教→[学长]等、其它PII→[脱敏]，老师本人保留） |
| 实验/决策记录 | 每阶段写入 docs/experiments.md |

## 3. 语料与评测集现状

### 3.1 语料（materials/）
| 类别 | 格式 | 文本层 | 处理 |
|---|---|---|---|
| 真题 2012–2026 | docx | ✅ 4.9 万字 | 直接解析 |
| 模拟卷（8 份+解析） | PDF | ✅ 文本 | 直接解析 |
| 专题一~十笔记 | PDF | ✅ 文本（含水印噪声） | 解析+去水印 |
| 讲义/背诵手册/学习指南 | docx/xlsx/pptx | ✅ | 直接解析 |
| 茆书题目难度分级 | PDF | ✅ 文本 | 直接解析 |
| **茆诗松教材（490 页）** | PDF | ❌ 纯扫描 | **按需 OCR（见 §7 路线）** |
| **贾书《统计学》第8版** | PDF | ❓ 疑似扫描 | **暂不处理** |

### 3.2 评测集（data/）
- `extracted_qa.json`：377 条知识点问答对（**确认型 97 / 硬 280**）
- `chapter_distribution.json`：280 条硬问答对的章节标注
- `待补录清单.json`：22 条（→ 状态「答案缺失」）
- `待复查清单.json`：30 条（空问答对，人工抽看）
- 智库.xlsx 清洗出的 7 条（待并入）

### 3.3 章节分布（280 条硬问答对，多模态补齐后）
- **数学基础（数三）152 条（54%）** — 检索统计教材无用，单列评测
- **统计专业课 127 条（45%）** — RAG 检索主战场
  - 高频：随机变量(27)、多维(22)、统计量(17)、描述统计(17)、回归(15)
  - 中低频：参数估计(10)、概率(9)、假设检验(9)、大数定律(1)

**启示**：①评测要按「数学基础 vs 统计专业课」分层，别混评；②茆书 OCR 优先级 = 高频统计章节；③数三数学是学生最大薄弱点。

## 4. 总体架构（RAG 全链路）

```
materials/ ──parse──> 纯文本+元数据 ──chunk──> 块 ──embed──> FAISS 索引
                                                              │
问题 ──retrieve──> top-k 块 ──prompt──> DeepSeek ──> 答案(+引用)
                                                              │
评测集 ──retrieve──> 块 ──生成──> 答案 vs gold ──> recall@k + LLM-as-judge
```

### 目录结构（MVP）
```
src/
  ingest/   parse.py chunk.py embed.py index.py   # 文档→索引
  rag/      retrieve.py generate.py               # 检索+生成
  eval/     metrics.py run_eval.py                # 评测
  utils/    config.py                             # 路径/key/常量
data/       # 评测集、解析产物、FAISS 索引（入库，无 PII）
docs/       # design.md experiments.md
```

## 5. 组件设计

### 5.1 文档解析（ingest/parse.py）
- PDF → PyMuPDF（`fitz`）；docx → python-docx；xlsx → openpyxl；pptx → python-pptx。
- 输出统一中间格式：`{source, page/位置, text, meta}`。
- 水印去噪：过滤「统计战线/就现在考研/微信公众号」等固定噪声串。
- 扫描 PDF（茆书）走 `extract.py` 的视觉 OCR（后续按需接入）。

### 5.2 分块（ingest/chunk.py）
- MVP：按「段落 + 固定长度（~512 字）+ 重叠（~64 字）」切分，保留 `source/page` 用于引用溯源。
- 公式题：暂不特殊处理，靠重叠保留上下文；badcase 需要再上结构化/公式感知分块。

### 5.3 Embedding + 索引（ingest/embed.py index.py）
- **BGE-M3**（本地，`FlagEmbedding`/`sentence-transformers`），dense 向量。
- **FAISS**（本地磁盘索引）。MVP 只做 dense top-k；混合检索（BM25）与重排（BGE-reranker）作为迭代项。

### 5.4 检索（rag/retrieve.py）
- `query → embed → FAISS top-k（默认 k=8）→ 返回块+来源`。
- 输出结构带来源（source/page），供生成时引用与评测 recall@k。

### 5.5 生成（rag/generate.py）
- Prompt：系统角色（中南 432 辅导老师）+ 检索到的 top-k 块 + 学生问题 → 要求「基于材料作答、标注引用来源、不会就说不会」。
- 模型：默认 `deepseek-flash`；识别到计算/推导题可切 `deepseek-v4-pro`（迭代项）。
- 生成层与模型解耦（换模型只改 config）。

### 5.6 评测（eval/）
- **检索指标**：recall@k（gold 答案所需材料是否在 top-k 中）、MRR。
- **答案指标**：LLM-as-judge（`deepseek-flash` 当裁判，对照 gold 答案打 1–5 分或 对/半对/错）。
- **分层**（关键）：
  - 统计专业课 127 条 → **主指标**（检索+答案）
  - 数学基础 152 条 → 单列（只评答案，不评检索）
  - 确认型 97 条 → 单列，不进主指标
  - 答案缺失 22 条 → 不进评测，仅真实问题分布统计

## 6. 评测集 schema（data/eval_set.json，待合并产出）

```json
{
  "id": "str",
  "问题": "str",
  "答案": "str | null",
  "来源": "真实 | 构造",
  "分类": "知识点 | 规划",
  "章节": "随机变量及其分布 | ... | 数学基础 | 其他",
  "确认型": false,
  "状态": "正式 | 答案缺失 | 待复查",
  "来源文件": "八月/IMG_4081.PNG | 智库.xlsx"
}
```
- 「答案缺失」：问题保留、答案 null，进分布统计、不进评测、**不删除**。
- 合并顺序：智库 7 条 + 截图 377 条 + 后续构造题。

## 7. 迭代路线（方案 C 的执行步骤）

1. **Step 1 基线**：文本版源 ingest + 朴素 RAG（BGE-M3 + FAISS dense top-k + DeepSeek）→ 跑出 recall@k + judge 基线。
2. **Step 2 评测分层**：按 §5.6 分层跑 280 条，定位统计专业课的 badcase。
3. **Step 3 badcase 分析**：哪些题该去茆书找但没找到 → **只 OCR 茆书相关章节**（优先：随机变量、多维、统计量、回归）→ 加入索引 → 复测。
4. **Step 4 增强**：按需加混合检索 / reranker / 公式分块（每个增强都用评测数据证明有效再留）。
5. 贾书 OCR：仅当 badcase 显示需要时再处理。
6. 每步结论写入 docs/experiments.md。

## 8. 风险与待定

- **BGE-M3 本地推理**：CPU 嵌入几百~几千块可行，速度待实测；过慢再评估 API 嵌入。
- **数学基础 54%**：检索无解，靠模型数学能力——是否值得单独引入数学题解法模块，待 Step 2 评测后定。
- **确认型/答案缺失占比**：主指标样本 = 统计专业课 127 条，量偏少，后续可构造补题（来源=构造）。
- **公式/LaTeX 处理**：茆书 OCR 与公式题检索质量是本项目已知难点，留到 badcase 驱动解决。
