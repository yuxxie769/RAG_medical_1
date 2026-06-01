# RAG Medical 技术方案

本文档记录项目第一阶段的当前实现方案，目标数据库为 **Milvus**，严格遵循《第一阶段开发计划与模块接口说明.md》中的范围：

- 仅实现 RAG 后端核心链路
- 先完成数据处理、知识库构建、检索召回、生成回答
- 暂不实现前端页面与复杂部署能力

---

## 1. 总体架构目标

第一阶段目标是完成一个**可追溯、低耦合、可扩展**的医疗问答 RAG 最小闭环。

核心链路如下：

```text
原始数据
→ 流式读取
→ 数据清洗
→ 知识库构建
→ 文档 embedding
→ Milvus 入库
→ 检索召回
→ LLM 生成回答
→ fallback / 日志 / 评估
```

设计原则：
- 模块职责单一
- 数据流单向传递
- 接口先行，避免耦合实现细节
- 批次可追踪，便于后续增量导入与排查
- 遇到单批错误时记录日志并跳过，不中断后续批次
- 支持断点恢复的导入状态设计（后续完善）

---

## 2. 模块分层

### 2.1 `app/data`
职责：
- 读取原始 `jsonl`
- 记录源文件路径与行号
- 清洗、去重、展开嵌套问题/答案
- 输出可供知识库构建使用的标准记录

要求：
- 支持流式读取大文件（主流程使用 `iter_raw_records`）
- 支持批次标识 `raw_batch_id`
- 保留 `source_path`、`source_line_no`
- 支持大文件按批次处理，不一次性全量载入内存

### 2.2 `app/kb`
职责：
- 将清洗后的记录转换为 `Document`
- 自动生成稳定的 `doc_id`
- 组装 `content`
- 保留源数据与批次元信息

要求：
- `doc_id` 只在知识库阶段生成
- `doc_id` 采用稳定哈希策略，保证同一条记录重复导入时主键一致
- `kb_batch_id` 继续保存在 `metadata` 中，用于追踪导入批次

### 2.3 `app/retrieval`
职责：
- dense 语义向量召回
- sparse/BM25 关键词召回
- Milvus 原生 RRF 混合召回
- 根据 API 参数切换 `hybrid`、`dense`、`sparse` 检索模式
- 返回标准检索结果 `RetrievalResult`

要求：
- 不包含生成逻辑
- 不直接依赖前端展示逻辑

### 2.4 `app/generation`
职责：
- 组装 prompt
- 调用 LLM
- 输出安全、谨慎的回答
- 无结果时 fallback

### 2.5 `app/evaluation`
职责：
- 计算 Recall@5 / Recall@10 / MRR
- 评估检索与生成效果

---

## 3. Milvus 作为目标向量数据库的设计

### 3.1 选择 Milvus 的原因
与当前项目目标匹配：
- 适合向量检索场景
- 支持持久化与索引管理
- 支持 BM25 Function、稀疏向量索引与原生混合检索
- Python SDK 完整，便于与 FastAPI / Python 服务集成

### 3.2 Milvus 在第一阶段的定位
Milvus 承担以下职责：
- 存储 dense embedding
- 根据 `content` 自动生成 sparse embedding
- 存储必要的文档元数据
- 提供 dense、sparse 和 hybrid TopK 查询
- 在 hybrid 模式下使用 RRF 完成两路结果融合

不在 Milvus 中实现：
- 业务逻辑
- 清洗逻辑
- 生成逻辑
- 评估逻辑

---

## 4. 文档与向量入库设计

### 4.1 `Document` 结构
知识库层统一使用 `Document`：

```text
Document
- doc_id
- question
- answer
- content
- metadata
```

其中 `source`、`split` 等辅助字段统一放入 `metadata`，避免主结构过重。

### 4.2 Milvus 存储字段
当前 collection 保存：

- `doc_id`：主键/唯一标识
- `embedding`：外部 embedding 模型生成的 dense 向量，类型为 `FLOAT_VECTOR`
- `sparse_embedding`：Milvus BM25 Function 自动生成的稀疏向量，类型为 `SPARSE_FLOAT_VECTOR`
- `content`：检索返回的文本主体，同时作为 BM25 Function 的输入
- `question`：问题文本
- `answer`：答案文本
- `source`：数据来源
- `split`：train/test/valid
- `clean_batch_id`、`raw_batch_id`、`source_path`、`source_line_no`、`kb_batch_id`、`kb_index`：追踪字段

### 4.3 向量字段策略
- dense 字段 `embedding` 由外部 embedding 模型生成，维度在项目配置中统一管理
- dense 索引使用 `HNSW`，相似度指标使用 `COSINE`
- `content` 启用中文 analyzer：`{"type": "chinese"}`
- sparse 字段 `sparse_embedding` 由 Milvus `FunctionType.BM25` 自动生成，应用层不手工计算或写入
- sparse 索引使用 `SPARSE_INVERTED_INDEX`，指标使用 `BM25`
- 旧版仅包含 dense 字段的 collection 不兼容当前 schema，必须清空重建

---

## 5. 数据流与批次管理

### 5.1 批次标识
为支持大规模导入与排查，统一采用分层批次标识：

- `raw_batch_id`：原始数据读取批次
- `clean_batch_id`：清洗批次
- `kb_batch_id`：知识库构建批次

### 5.2 `doc_id` 生成规则
`doc_id` 在 `kb` 层自动生成，不需要人工手写。当前采用稳定哈希策略：

```text
sha256(question + answer + source_path + source_line_no)
```

输出为 64 位十六进制字符串，并作为 Milvus 的 `VARCHAR` 主键使用。

优势：
- 同一条清洗记录重复导入时生成相同 `doc_id`
- 不依赖 `kb_batch_id`，避免不同导入批次产生重复文档
- 保持 Milvus schema 不变，无需启用 `auto_id`
- 仍可通过 `metadata.kb_batch_id`、`metadata.kb_index` 追踪导入批次与批内顺序

注意：
- 如果同一问答内容来自不同源文件或不同行号，会生成不同 `doc_id`
- 如果需要跨来源强去重，可后续将哈希输入调整为仅 `question + answer`

---

## 6. 向量化与入库流程

推荐流程如下：

1. 从 `app/data` 以流式方式获取清洗结果
2. `app/kb` 转换为 `Document`
3. `embedder` 按批次将 `content` 转为 dense embedding
4. 按批次将 dense embedding、`content` 与元数据写入 Milvus
5. Milvus 根据 `content` 自动生成 `sparse_embedding`
6. 保存索引版本和批次信息

### 6.1 入库建议
- 主流程采用流式读取，不一次性加载全量数据
- 向量化采用分批处理，减少内存压力
- 每批入库完成后记录日志
- 单批失败时记录错误并跳过，不中断后续批次
- 重复 `doc_id` 直接报错，避免污染库
- 保留输入批次与 Milvus 主键映射
- 导入任务需要有独立审计日志与恢复状态，后续补齐
- collection 重建后重新导入相同文件时，需要传入 `force_reingest=true`，忽略旧断点状态

### 6.2 Collection 重建
当前 hybrid schema 与旧版 dense-only schema 不兼容。服务启动时如果检测到缺少 `sparse_embedding` 或 BM25 Function，会返回明确错误，不会自动删除旧数据。

确认不需要保留旧数据后，执行：

```bash
python scripts/reset_milvus_collection.py --confirm-drop
```

脚本会删除同名 collection，并按照当前 schema 创建空 collection。随后重新调用 `POST /ingest`：

```json
{
  "source_path": "tests/sample_data.jsonl",
  "force_reingest": true
}
```

`--confirm-drop` 是必需的显式保护参数。未传入时，如果 collection 已存在，脚本会拒绝删除。

---

## 7. 检索接口设计

### 7.1 检索输入
- `query`：用户问题
- `search_method`：搜索方法，可选值为 `hybrid`、`dense`、`sparse`，默认 `hybrid`
- `top_k`：最终返回条数
- `fetch_k`：向 Milvus 请求的候选条数，默认不小于 `top_k`
- `min_score`：最低过滤分数
- 预留过滤条件（当前 API 尚未开放）：
  - `source`
  - `split`
  - 其他 metadata 条件

### 7.2 检索输出
返回标准结构 `RetrievalResult`：

```text
RetrievalResult
- doc_id
- content
- score
- metadata
```

### 7.3 统一检索流程
`POST /query` 的统一处理流程：

```text
请求参数
→ 解析 search_method / top_k / fetch_k / min_score
→ 确认 Milvus collection 已建立且非空
→ 按 search_method 执行 dense、sparse 或 hybrid 召回
→ 从 Milvus 获取按 score 降序排列的 fetch_k 个候选
→ 应用层按 min_score 过滤
→ 截断为 top_k 条结果
→ 返回 RetrievalResult[]
```

`fetch_k` 控制召回候选规模，`top_k` 控制最终输出规模。当前默认值分别为 `15` 和 `5`。

### 7.4 Dense 召回逻辑与算分
当 `search_method="dense"` 时：

```text
query
→ 外部 embedding 模型生成 query embedding
→ Milvus 在 embedding 字段执行 HNSW 检索
→ 使用 COSINE 相似度排序
→ 返回 dense score
```

COSINE 相似度公式：

```text
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

`score` 表示 query 向量与文档向量的余弦相似度。分数越大，语义越接近。

### 7.5 Sparse/BM25 召回逻辑与算分
当 `search_method="sparse"` 时：

```text
query 原文
→ Milvus 中文 analyzer 分词
→ 在 sparse_embedding 字段执行 BM25 检索
→ 返回 BM25 score
```

入库时，Milvus 已通过 BM25 Function 将 `content` 自动转换为 `sparse_embedding`。查询时应用层直接传入 query 原文，不调用外部 embedding 模型。

BM25 综合考虑词频、逆文档频率和文档长度归一化。`score` 越大，关键词匹配越强。BM25 分数没有固定的 `0~1` 范围。

### 7.6 Hybrid 召回逻辑与算分
当 `search_method="hybrid"` 时：

```text
query
├─→ 外部 embedding 模型生成 query embedding
│   → embedding 字段执行 COSINE dense 检索
└─→ query 原文
    → sparse_embedding 字段执行 BM25 检索

dense 候选 + sparse 候选
→ Milvus RRFRanker(k=60)
→ 返回融合后的 RRF score
```

应用层创建两个 `AnnSearchRequest`，每路最多召回 `fetch_k` 条候选，再由 Milvus 内置 `RRFRanker(k=60)` 融合。RRF 只使用文档在各路召回中的名次，不直接混加 COSINE 与 BM25 原始分数：

```text
RRF_score(document) = Σ 1 / (k + rank_i(document))
```

- `k` 当前配置为 `60`
- `rank_i(document)` 表示文档在第 `i` 路召回中的排名
- 如果文档没有进入某一路候选集，该路不贡献分数
- 文档在 dense 与 sparse 两路的排名越靠前，最终 RRF score 越高

示例：某文档在 dense 和 sparse 两路都排名第 1：

```text
1 / (60 + 1) + 1 / (60 + 1) ≈ 0.0327869
```

因此，hybrid 返回的 `0.0325` 左右分数属于正常的 RRF 排序分数，不表示 `3.25%` 相关度。

### 7.7 Score 使用约束
三种模式返回的 `score` 含义不同：

| `search_method` | `score` 含义 | 是否可与其他模式直接比较 |
| --- | --- | --- |
| `dense` | COSINE 相似度 | 否 |
| `sparse` | BM25 关键词匹配分数 | 否 |
| `hybrid` | RRF 排名融合分数 | 否 |

当前 `min_score` 默认值为 `0.0`。如果后续启用更高阈值，必须按检索模式分别基于评测集校准，不能将 dense 阈值直接复用于 hybrid 或 sparse。

### 7.8 预留方向
- 后续可以增加 query 改写、路由策略和多轮召回
- 当前 RRF 属于召回融合，不等同于 cross-encoder 或 LLM 精排；如需更高精度，可在候选集之后增加独立 rerank 阶段

---

## 8. 生成阶段与安全策略

由于项目是医疗问答场景，生成阶段必须默认谨慎。

### 8.1 生成输入
- 用户问题 `query`
- 检索结果 `RetrievalResult[]`

### 8.2 生成输出
`GenerationResult`：
- `query`
- `answer`
- `citations`
- `fallback`
- `latency`

### 8.3 安全策略
- 检索不到时直接 fallback
- 高风险医疗建议保持克制
- 不输出超出证据范围的结论
- 必要时提示“建议就医 / 进一步检查”

---

## 9. 评估设计

### 9.1 检索评估
- Recall@5
- Recall@10
- MRR

### 9.2 生成评估
- 相关性
- 完整性
- 忠实性
- 安全性

### 9.3 评估原则
- 评估与主流程分离
- 先做程序化评估，再做 LLM-as-Judge
- 记录评估样本与版本号

---

## 10. 当前 API 接口定义

### 10.1 `POST /ingest`
导入原始数据、清洗、构建知识库并写入 Milvus。

#### 输入
```json
{
  "source_path": "tests/sample_data.jsonl",
  "raw_batch_id": "raw_001",
  "clean_batch_id": "clean_001",
  "kb_batch_id": "kb_001",
  "flush_to_milvus": true,
  "force_reingest": false
}
```

#### 字段说明
- `source_path`：原始 `jsonl` 文件路径，必填。支持项目外绝对路径。
- `raw_batch_id`：原始数据批次，可选
- `clean_batch_id`：清洗批次，可选
- `kb_batch_id`：知识库批次，可选
- `flush_to_milvus`：是否写入 Milvus，默认 `true`
- `force_reingest`：是否忽略相同来源文件的历史断点状态，默认 `false`。清空并重建 collection 后重新导入时设置为 `true`

#### 输出
```json
{
  "status": "ok",
  "source_path": "tests/sample_data.jsonl",
  "raw_count": 10,
  "cleaned_count": 11,
  "document_count": 11,
  "milvus_written": true,
  "milvus_mode": "milvus"
}
```

#### 字段说明
- `status`：处理状态
- `source_path`：输入文件路径
- `raw_count`：原始读取条数
- `cleaned_count`：清洗后条数
- `document_count`：构建出的文档数
- `milvus_written`：是否写入成功
- `milvus_mode`：当前存储模式，固定为 `milvus`

---

### 10.2 `POST /query`
返回检索结果，不做生成。

#### 输入
```json
{
  "query": "口干的治疗方案是什么？",
  "search_method": "hybrid",
  "top_k": 5,
  "fetch_k": 15,
  "min_score": 0.0
}
```

#### 字段说明
- `query`：用户查询文本，必填
- `search_method`：检索模式，支持 `hybrid`、`dense`、`sparse`，默认 `hybrid`
- `top_k`：最终返回条数，默认 `settings.top_k`
- `fetch_k`：召回候选条数，默认 `settings.fetch_k`
- `min_score`：最低分过滤阈值，默认 `settings.min_score`

#### 输出
```json
{
  "query": "口干的治疗方案是什么？",
  "search_method": "hybrid",
  "top_k": 5,
  "fetch_k": 15,
  "min_score": 0.0,
  "hits": [
    {
      "doc_id": "...",
      "content": "Q: ...\nA: ...",
      "score": 0.032522473484277725,
      "metadata": {}
    }
  ]
}
```

#### 字段说明
- `search_method`：实际使用的检索模式
- `hits`：检索结果列表，按当前模式的分数降序返回
- `hits[].score`：由当前检索模式决定。示例中的 `0.0325` 是 hybrid 模式下的 RRF 分数

---

### 10.3 `POST /answer`
检索 + LLM 生成最终答案。

#### 输入
```json
{
  "query": "口干的治疗方案是什么？",
  "top_k": 5
}
```

#### 输出
```json
{
  "query": "口干的治疗方案是什么？",
  "answer": "...",
  "fallback": false,
  "citations": [
    {
      "doc_id": "...",
      "content": "Q: ...\nA: ...",
      "score": 0.032522473484277725,
      "metadata": {}
    }
  ]
}
```

#### 字段说明
- `answer`：最终生成结果
- `fallback`：是否走了降级回答
- `citations`：生成使用的检索证据

---

## 11. 配置与扩展点

### 11.1 建议配置项
- Milvus 连接信息
- collection 名称
- embedding 模型名称
- embedding 维度
- LLM 模型名称
- top_k / fetch_k / min_score
- search_method
- Milvus dense 指标、sparse 字段名、中文 analyzer 类型和 RRF 参数
- batch size
- 重试次数
- 导入审计日志路径
- 导入状态持久化路径

### 11.2 当前混合检索配置
- `SEARCH_METHOD=hybrid`
- `MILVUS_DENSE_METRIC_TYPE=COSINE`
- `MILVUS_SPARSE_FIELD_NAME=sparse_embedding`
- `MILVUS_ANALYZER_TYPE=chinese`
- `HYBRID_RRF_K=60`

### 11.3 未来扩展点
- cross-encoder 或 LLM 精排
- query 改写
- 查询路由
- 本地模型替换 API 模型
- 导入断点续跑

---

## 12. 第一阶段验收标准

第一阶段完成时应满足：
1. 能将清洗后的数据构建为 `Document`
2. 能将 `Document` 向量化并写入 Milvus
3. 能基于 query 从 Milvus 返回 TopK 结果
4. 能将检索结果传递给生成模块
5. 能完成无结果 fallback
6. 能记录批次、行号、doc_id 与异常信息
7. 能输出基础检索指标
8. 能记录导入审计日志
9. 导入中断后可从状态继续恢复

---

## 13. 当前实现状态与后续顺序

当前已完成：

1. 向量化接口与 Milvus 入库
2. Milvus dense + BM25 sparse collection schema
3. 流式读取、分批 embedding 与导入状态记录
4. `hybrid`、`dense`、`sparse` 三种召回模式
5. Milvus 原生 RRF 融合召回

后续推荐继续：

1. 补充模式级检索评测与阈值校准
2. 增加独立精排阶段
3. 完善生成安全策略和评估流程

---

## 14. 备注

本文件是项目第一阶段的技术档案，后续随着实现推进会继续补充具体接口、字段与参数定义。
