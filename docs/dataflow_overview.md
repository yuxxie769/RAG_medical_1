# 数据流 × ID 体系 × 幂等设计 × Ingest

本文档统一梳理 RAG_medical 项目中一条 JSONL 行从磁盘到最终答案的完整流程，主要涵盖数据流、ID 体系、幂等设计和导入系统四个主要技术亮点。

---

## 1. 数据流全景：一条 JSONL 行的完整流程

```
┌─ JSONL 磁盘文件 ───────────────────────────────────────────────────────┐
│  {"question": "...", "answer": "..."}                                  │
└──┬────────────────────────────────────────────────────────────────────┘
   │ app/data/loader.py  iter_raw_records()
   │ · 枚举行号 source_line_no（1-based）
   │ · 挂载 source_path（绝对路径）、raw_batch_id（文件名 stem）
   ▼
┌─ 原始记录 ─────────────────────────────────────────────────────────────┐
│  { raw_batch_id, source_path, source_line_no, record: {...} }         │
└──┬────────────────────────────────────────────────────────────────────┘
   │ app/data/cleaner.py  clean_records()
   │ · 展平嵌套问题/答案、去重、标准化
   │ · 透传 raw_batch_id，追加 clean_batch_id
   ▼
┌─ 清洗记录 ─────────────────────────────────────────────────────────────┐
│  { question, answer, source_path, source_line_no,                     │
│    raw_batch_id, clean_batch_id, meta: {...} }                        │
└──┬────────────────────────────────────────────────────────────────────┘
   │ app/kb/builder.py  build_documents()
   │ · 生成确定性 doc_id（SHA256）
   │ · 长答案(>1648字符) 滑动窗口分块
   │ · 挂载 kb_batch_id、kb_index
   ▼
┌─ Document 对象 ────────────────────────────────────────────────────────┐
│  { doc_id, question, answer, content("Q:...\nA:..."), metadata }      │
└──┬────────────────────────────────────────────────────────────────────┘
   │ app/retrieval/indexer.py  VectorIndexer.index_in_batches()
   │ · embedder.embed_texts([doc.content]) → 1024维 FLOAT_VECTOR
   │ · build_records(docs, vectors) → MilvusRecord[]
   ▼
┌─ Milvus ───────────────────────────────────────────────────────────────┐
│  主键 = doc_id (VARCHAR 256, auto_id=False)                            │
│  Dense: HNSW + COSINE  │  Sparse: BM25 Function (中文 analyzer)        │
│  upsert 前查重 _existing_doc_ids() → 只插入缺失记录                     │
│  upsert 后 verify_persisted_doc_ids() → 回读校验                        │
└──┬────────────────────────────────────────────────────────────────────┘
   │ POST /query  三种检索模式
   ▼
┌─ 检索召回 ─────────────────────────────────────────────────────────────┐
│  hybrid: RRF(k=60) 融合 dense COSINE + sparse BM25                     │
│  dense : HNSW + 余弦相似度                                              │
│  sparse: 中文分词 + BM25 关键词匹配                                      │
│  返回 RetrievalResult[] → POST /answer → LLM 生成 → 最终答案           │
└─────────────────────────────────────────────────────────────────────────┘
```



---

## 2. ID 体系：两层设计

### 2.1 业务主键层 — 确定性哈希

```
doc_id = SHA256(
    question + "\n" +
    answer   + "\n" +
    source_path + "\n" +
    str(source_line_no)
)
```

`app/kb/builder.py:35` — 四元组输入，64 字符 hex 输出。

**核心特性：**

| 特性 | 说明 |
|------|------|
| 确定性 | 相同四元组永远产出相同 doc_id，不依赖运行环境 |
| 唯一性 | SHA256 碰撞概率 < 2⁻²⁵⁶，不同数据行必得不同 ID |
| 来源绑定 | question+answer 相同但不同文件/行号 → 不同 doc_id（防止误合并） |
| 幂等性基石 | 重新导入同数据不产生新 ID，upsert 可自动跳过已有文档 |

### 2.2 分块时的 ID 派生

```
长答案分块后:
  parent_doc_id = SHA256(question + answer + source_path + source_line_no)  # 不分块时也是它
  chunk_doc_id  = SHA256(parent_doc_id + "\n" + chunk_index)                # 分块时每个 chunk 专属
```

`app/kb/builder.py:48` — chunk 层级 ID 同样确定可算，幂等性穿透到 chunk。

分块后 `metadata` 追加字段：
- `is_chunked`, `parent_doc_id`, `chunk_index`, `chunk_count`
- `chunk_start`, `chunk_end`, `original_answer_length`

检索命中 chunk 后可通过 `parent_doc_id` 找到同文档的所有 chunk。

### 2.3 运行实例层 — 随机 UUID4

| ID | 生成方式 | 用途 |
|----|---------|------|
| `ingest_run_id` | `uuid4().hex`（32 字符） | 一次导入的全局标识 |
| `batch_id` | `uuid4().hex`（32 字符） | 一批 N 行的处理单元 |
| `kb_batch_id` | `uuid4().hex[:8]`（8 字符） | 知识库构建批次，透传至 Milvus metadata |

### 2.4 溯源链路

```
ingest_run_id ──→ ingest_batches ──→ kb_batch_id ──→ Milvus metadata
                    │
                    └──→ ingest_events（审计）
```

`kb_batch_id` 存入 Milvus 每条记录的 metadata，可从向量反向追溯：
- 哪个 batch 写入的（`ingest_batches.batch_id`）
- 哪次 ingest run（`ingest_batches.ingest_run_id`）
- 原始文件哪一行（`source_path + source_line_no`）

---

## 3. 幂等设计：三层保障

### 3.1 第 1 层 — 写入前查重


已存在的 doc_id 不再写入，返回 `existing_count` 供上层审计。

### 3.2 第 2 层 — 写入后回读验证

所有批次 flush 后，InjstService 调用此方法二次确认数据真正落盘：

### 3.3 第 3 层 — 批次级断点续跑

`ingest_batches` 集合以 `(ingest_run_id, start_line, end_line)` 唯一索引记录每批状态。重启导入时：
- `succeeded` → 跳过
- `pending` / `failed` → 重试
- `running`（租约过期残留）→ 重置为 `pending` 后重试

**三层覆盖的场景**：

| 中断时机 | 覆盖层 | 行为 |
|----------|--------|------|
| embedding 前 | 第 3 层 | 批次标记 pending，重试 |
| Milvus 写入前 | 第 1 层 | 查重发现已存在（如果之前部分写入成功），只补缺 |
| Milvus 写入后、MongoDB 更新前 | 第 1+2 层 | 查重跳过已有 doc_id，校验通过后 MongoDB 安全标记成功 |
| MongoDB 写入后 | 第 3 层 | succeeded 批次直接跳过 |


---

## 4. 分块逻辑

位置：`app/kb/builder.py`

### 4.1 参数

```python
ANSWER_CHUNK_THRESHOLD = 1648  # 超过此字符数才分块
ANSWER_CHUNK_SIZE      = 700   # 每个 chunk 的字符数
ANSWER_CHUNK_OVERLAP   = 70    # 相邻 chunk 重叠字符数
ANSWER_CHUNK_STEP      = 630   # 滑动步长 = SIZE - OVERLAP
```

### 4.2 规则

```
len(answer) ≤ 1648 → 不分块，1 条 Document，doc_id = parent_doc_id
len(answer) > 1648 → 滑动窗口分块，N 条 Document，doc_id = chunk_doc_id
```

```
answer: "ABCDEFGHIJKLMNOPQRSTUVWXYZ..." (len=2000)
         └── CHUNK_0 [0..700)   ────┘
                  └── CHUNK_1 [630..1330) ────┘
                           └── CHUNK_2 [1260..1960) ──┘
                                    └── CHUNK_3 [1890..2000) ┘ (尾块)
```

### 4.3 设计要点

| 要点 | 说明 |
|------|------|
| 仅切 answer | question 不分块，每个 chunk 的 content 拼成 `"Q: {q}\nA: {chunk}"` |
| question 完整保留 | 每个 chunk 都存储完整 question，检索时 Q 信息不丢失 |
| 确定性 chunk ID | 分块后幂等性无损 |
| parent_doc_id 穿透 | 可通过 metadata 找到同父文档的所有 chunk |

### 4.4 当前局限

- 纯字符数切割，不感知句子/段落边界

---

## 5. Ingest 导入系统

### 5.1 架构概览

```
POST /ingest ─→ 校验路径 + MongoDB Ping
              → 获取/接管租约（409 如果冲突）
              → 创建后台任务
              → 202 Accepted（立即返回）
                    │
    ┌───────────────┘
    ▼
后台执行:
  count_total_lines()                # 估算 total_lines
  → 流式读 JSONL (iter_raw_records)
  → 按 batch_size 组批
  → 每批: get_or_create_batch()
           ├ succeeded? → 跳过
           └ pending/failed? → clean → build → embed → upsert → mark_succeeded
  → flush + verify
  → finalize_run
```

### 5.2 MongoDB 相关log集合

| 集合 | 用途 |
|------|------|
| `ingest_runs` | 任务主记录：状态、租约、checkpoint_line、汇总统计 |
| `ingest_batches` | 行区间批次：状态、attempt_count、写入计数 |
| `ingest_events` | 追加审计日志：创建/恢复/成功/失败/熔断/完成 |

### 5.3 状态机

**Run 状态（`status` 字段）：**

```
queued ──→ running ──→ completed
  │          │  │        completed_with_errors
  │          │  └──────→ failed
  │          │
  └──────────┼──→ cancelling ──→ cancelled
             │
             └──→ cancelled（执行前即取消）
```

| 状态 | 含义 | 触发 |
|------|------|------|
| `queued` | 任务已创建，等待后台执行 | `start_run()` |
| `running` | 后台正在执行 | `mark_run_running()` |
| `completed` | 全部批次成功，无失败缺口 | `finalize_run()` |
| `completed_with_errors` | 扫描结束，但仍有失败批次 | `finalize_run()` |
| `failed` | 顶层异常或熔断触发 | `fail_run()` / 熔断 |
| `cancelling` | 已收到取消请求，等待当前批次结束 | `request_cancel()` |
| `cancelled` | 取消完成 | `mark_run_cancelled()` |

**阶段（`stage` 字段）：** 细化 `running` stage的内部进度：
```
queued → counting_lines → ingesting → finished
```
`cancelling` 期间 `stage=cancelling`，终态统一为 `stage=finished`。

**批次状态：**

```
pending ─→ running ─→ succeeded
  ↑                    failed
  └── 租约过期后被接管时，running 重置为 pending
```

### 5.4 run任务租约

同一个 `(source_path, collection_name)` 只有一个活跃租约：

| 属性 | 值 |
|------|-----|
| 粒度 | `(source_path, collection_name)` |
| 默认时长 | 300 秒（`INGEST_LEASE_SECONDS`） |
| 冲突行为 | 返回 409 |
| 过期恢复 | 新请求接管原 run，遗留 `running` 批次重置为 `pending` |

正常流程每批开始前续租（`renew_lease`），完成或异常后释放租约。

### 5.5 run任务返回字段checkpoint_line vs last_scanned_line

```
例：第 1-100 行成功，第 101-200 行失败，第 201-300 行成功
  last_scanned_line = 300   （"扫到第几行"，不管成功与否）
  checkpoint_line   = 100   （"连续成功的最远行"，不跨越失败缺口）
```

`checkpoint_line` 由 `_summarize()` 按 `start_line` 升序遍历批次计算：遇非 `succeeded` 批次即停止推进。

### 5.6 熔断

连续 3 次 embedding 或 Milvus 写入失败 → 停止导入：

| 阶段 | 行为 |
|------|------|
| 第 1-2 次 | 记录 `batch_failed`，继续处理后续批次 |
| 第 3 次 | 记录失败批次 → 写入 `circuit_breaker_opened` 事件 → run 标记 `failed` → 释放租约 → 停止扫描 |

触发条件仅限 `embedding` 和 `milvus` 两类基础设施故障（API 429/鉴权失败/连接断开/写入失败/校验失败）。清洗和构建异常不触发熔断，仅标记单批次失败。

基础设施恢复后再次调用 `POST /ingest` 即可续跑。

### 5.7 API 接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/ingest` | POST | 异步提交导入任务 → 202 |
| `/ingest/status?ingest_run_id=` | GET | 查询状态、进度、失败详情 |
| `/ingest/{run_id}/events` | GET | SSE 实时进度流 |
| `/ingest/{run_id}` | DELETE | 协作取消 |

### 5.8 SSE 实时进度流

`GET /ingest/{ingest_run_id}/events` 返回 `text/event-stream`，替代轮询，实时推送导入进度。

**行为：**

```
客户端连接
  → 立即推送当前快照（progress 事件）
  → 轮询 MongoDB（1 秒间隔）
     ├ updated_at 变化 → 推送 progress 事件（完整 IngestStatusResponse JSON）
     └ 15 秒无变化  → 发送心跳注释 ": heartbeat\n\n"
  → run 到达终态（completed / completed_with_errors / failed / cancelled）
     → 推送最后一次 progress 事件
     → 关闭连接
  → MongoDB 不可用
     → 推送 error 事件（含终端 failed 状态的快照）
     → 关闭连接
```

实现位于 `app/api/main.py:358`，`_format_sse()` 按 SSE 规范组装 `event:` 和 `data:` 行。

**事件类型：**

| 事件 | 触发条件 | 内容 |
|------|---------|------|
| `progress` | `updated_at` 变化（初次连接、每批完成、阶段切换） | `IngestStatusResponse` 完整 JSON |
| `error` | MongoDB 轮询失败 | 简化快照，status=failed |
| `: heartbeat` | 15 秒无变化 | SSE 注释行，保持连接 |

连接不再需要手动关闭，也不需客户端显式断开。

### 5.9 关键边界场景

| 场景 | 行为 |
|------|------|
| 同文件+同collection 重复导入 | 复用 run，跳过 succeeded 批次，重试 failed/pending，Milvus 幂等去重 |
| 同文件+换 collection | 新建 run，重新全量导入 |
| 不同文件 | 完全隔离，独立 run |
| 文件内容变了但路径不变 | 不自动识别，需 `force_reingest=true` 或改名 |
| Milvus 已写但 MongoDB 未记 | 3.3 节三层幂等保障覆盖 |
| MongoDB 不可用 | `/ingest` 返回 503，拒绝导入 |
| 两个请求同时导同目标 | 租约冲突，后者返回 409 |
| 两个请求同时导不同目标 | 各自独立，不冲突 |
