# RAG Medical Architecture

本文档描述当前代码仓库里已经落地的后端架构，而不是早期设想。

当前系统的目标是提供一条完整的医疗 RAG 后端链路：

- JSONL 数据导入
- 清洗与知识库文档构建
- dense + sparse + hybrid 检索
- 融合召回后的可插拔 rerank
- LLM 生成回答
- MongoDB 导入状态、断点续跑、审计与进度跟踪

---

## 1. Overall Flow

### 1.1 Ingest flow

```text
source jsonl
-> stream read
-> clean records
-> build Document
-> embed content
-> write / verify Milvus
-> persist run + batch state in MongoDB
```

### 1.2 Query flow

```text
query
-> dense / sparse / hybrid recall
-> min_score filter on retrieval score
-> optional rerank
-> top_k truncate
-> /query response
-> /answer reuses query results as citations
```

---

## 2. Runtime Components

### 2.1 API layer

Entry: [app/api/main.py](C:/Users/xieyuxiang/Documents/RAG_medical/app/api/main.py)

Responsibilities:

- expose `POST /ingest`
- expose `GET /ingest/status`
- expose `GET /ingest/{ingest_run_id}/events` as SSE
- expose `DELETE /ingest/{ingest_run_id}`
- expose `POST /query` and `POST /answer`
- build and cache runtime dependencies:
  - embedder
  - Milvus store
  - retriever
  - reranker
  - generator
  - ingest repository/service

### 2.2 Data layer

Modules under `app/data` are responsible for:

- streaming JSONL input
- preserving `source_path` and `source_line_no`
- cleaning and normalizing records
- expanding source rows into cleaned question-answer records

The ingest path is stream-oriented and does not load the whole file into memory.

### 2.3 KB layer

Modules under `app/kb` convert cleaned records into `Document`.

Current behavior:

- `content` is assembled from question + answer
- stable `doc_id` is generated in KB layer
- answer chunking is handled in KB build path
- document metadata carries batch and source tracing fields

### 2.4 Retrieval layer

Modules under `app/retrieval` provide:

- external dense embedding
- Milvus-backed dense retrieval
- Milvus BM25 sparse retrieval
- Milvus native hybrid retrieval with `RRFRanker`
- post-recall rerank

Important boundary:

- Milvus handles recall and dense/sparse fusion
- application layer handles rerank
- generation logic is not mixed into retrieval

### 2.5 Generation layer

Modules under `app/generation` build prompts and call an OpenAI-compatible chat completion API.

`/answer` currently works as:

```text
/answer
-> internally call query flow
-> use returned hits as citations
-> generate final answer
```

### 2.6 Ingest state layer

Modules under `app/ingest` use MongoDB as the only source of truth for ingest state.

Collections:

- `ingest_runs`
- `ingest_batches`
- `ingest_events`

This layer owns:

- leases
- resumable progress
- batch state
- audit trail
- cancellation flags
- stage and progress metrics

---

## 3. Storage Design

### 3.1 Milvus

Milvus is the vector and retrieval store.

Current collection stores:

- `doc_id`
- `embedding`
- `sparse_embedding`
- `content`
- `question`
- `answer`
- metadata fields such as:
  - `source`
  - `split`
  - `raw_batch_id`
  - `clean_batch_id`
  - `kb_batch_id`
  - `source_path`
  - `source_line_no`

Current retrieval behavior:

- `dense`: vector search on `embedding`
- `sparse`: BM25 search on `sparse_embedding`
- `hybrid`: Milvus native `hybrid_search` + `RRFRanker`

Milvus write semantics are idempotent by stable `doc_id`:

- existing `doc_id` is treated as already written
- only missing docs are inserted
- post-write verification confirms expected `doc_id`s exist

### 3.2 MongoDB

MongoDB stores ingest execution state and audit records.

`ingest_runs` includes:

- lifecycle status
- stage
- lease fields
- source path and collection name
- aggregate counts
- checkpoint and scan position
- cancel flags
- progress metrics

`ingest_batches` includes:

- line ranges
- batch status
- attempt count
- per-batch counters
- per-batch error

`ingest_events` is append-only and records:

- run queued / started / resumed / finished
- batch succeeded / failed / skipped
- cancel requested
- circuit breaker opened

---

## 4. Ingest Architecture

### 4.1 Execution model

`POST /ingest` is asynchronous.

Request behavior:

1. validate source path
2. check Mongo availability
3. create or resume run
4. acquire lease
5. schedule background execution
6. return `202 Accepted` with:
   - `ingest_run_id`
   - `status_url`
   - `events_url`
   - `cancel_url`

### 4.2 Run state model

`status` describes lifecycle outcome:

- `queued`
- `running`
- `cancelling`
- `cancelled`
- `completed`
- `completed_with_errors`
- `failed`

`stage` describes what the worker is currently doing:

- `queued`
- `counting_lines`
- `ingesting`
- `cancelling`
- `finished`

Why both exist:

- `status` answers "what is the task state"
- `stage` answers "what is it doing right now"

### 4.3 Lease model

Lease fields:

- `lease_owner`
- `lease_expires_at`

Purpose:

- prevent two requests from ingesting the same `(source_path, collection_name)` at the same time
- make forced process exit recoverable

Behavior:

- active lease -> new ingest returns `409`
- expired lease -> next normal ingest can resume
- stale `running` batches are reset to `pending`

### 4.4 Resume semantics

Resume is automatic for the same `(source_path, collection_name)`.

Rules:

- succeeded batches are skipped
- failed batches are retried
- checkpoint only advances across contiguous succeeded batches
- `last_scanned_line` tracks how far the worker scanned
- recovery is based on `ingest_batches`, not only run-level summary counters

### 4.5 Progress and SSE

Current progress model includes:

- `total_lines`
- `current_scanned_line`
- `progress_percent`
- `processed_batches`
- `skipped_succeeded_batches`
- `elapsed_seconds`
- `lines_per_second`
- `batches_per_second`
- `embedding_seconds_total`
- `embedding_seconds_last_batch`
- `eta_seconds`

SSE endpoint:

```text
GET /ingest/{ingest_run_id}/events
```

Behavior:

- send snapshot immediately
- push on status update
- send heartbeat when idle
- close on terminal state

### 4.6 Cancellation

Cancellation is cooperative.

`DELETE /ingest/{ingest_run_id}` sets:

- `cancel_requested = true`
- `status = cancelling`
- `stage = cancelling`

Worker behavior:

- checks cancel flag between batches
- finishes the current batch safely
- stops before the next batch
- marks run `cancelled`
- releases lease

### 4.7 Batch logging

Every batch completion is logged and audited:

- `batch_succeeded`
- `batch_failed`
- `batch_skipped`

Each log/event includes enough context for troubleshooting:

- run id
- batch id
- line range
- attempt count
- counts
- timing
- error

### 4.8 Infrastructure circuit breaker

Current implementation uses a simple infrastructure circuit breaker.

If 3 consecutive batches fail due to infrastructure failures, the run stops early.

Infrastructure failures currently include:

- `embedding`
  - quota exhausted
  - 429
  - auth failure
  - upstream embedding service unavailable
- `milvus`
  - collection check/create failure
  - connection failure
  - write failure
  - persistence verification failure

Non-infrastructure failures such as:

- cleaning errors
- KB build errors
- bad record content

do not open the circuit breaker; they remain normal failed batches.

---

## 5. Retrieval and Rerank Architecture

### 5.1 Recall modes

Current request modes:

- `dense`
- `sparse`
- `hybrid`

`dense`:

- embed query
- search vector field

`sparse`:

- search BM25 sparse field
- does not call dense embedder

`hybrid`:

- embed query
- execute dense + sparse recall
- let Milvus fuse with `RRFRanker`

### 5.2 Query pipeline

Current `/query` path:

1. resolve `fetch_k`
2. recall candidates from Milvus
3. apply `min_score` using original retrieval score
4. run rerank if enabled
5. truncate to `top_k`

This means `min_score` semantics stay backward-compatible.

### 5.3 Rerank stage

Rerank is application-level and pluggable.

Current providers:

- `none`
- `openai_compatible`

Provider selection is controlled by environment variables only.

Current config fields:

- `ENABLE_RERANK`
- `RERANK_PROVIDER`
- `RERANK_API_KEY`
- `RERANK_BASE_URL`
- `RERANK_PATH`
- `RERANK_MODEL_NAME`
- `RERANK_TIMEOUT_SECONDS`
- `RERANK_CANDIDATE_LIMIT`

### 5.4 OpenAI-compatible rerank contract

Current expected request:

```json
{
  "model": "...",
  "query": "...",
  "documents": ["doc text 1", "doc text 2"],
  "top_n": 15
}
```

Current expected response:

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.98},
    {"index": 1, "relevance_score": 0.75}
  ]
}
```

If rerank fails due to timeout, 429, quota, 5xx, or malformed response:

- log warning
- fall back to original recall order
- keep `/query` and `/answer` successful

### 5.5 Response metadata

Each hit may include:

- `retrieval_score`
- `rerank_score`
- `rerank_applied`
- `rerank_provider`

Current meaning of `score`:

- rerank applied -> final rerank score
- rerank not applied -> original retrieval score

---

## 6. Public API Snapshot

### 6.1 Ingest

- `POST /ingest`
- `GET /ingest/status`
- `GET /ingest/{ingest_run_id}/events`
- `DELETE /ingest/{ingest_run_id}`

### 6.2 Retrieval

`POST /query`

Input:

- `query`
- `search_method`
- `top_k`
- `fetch_k`
- `min_score`

Output:

- `query`
- `search_method`
- `top_k`
- `fetch_k`
- `min_score`
- `hits[]`

### 6.3 Answer generation

`POST /answer`

Input:

- `query`
- `top_k`

Output:

- `query`
- `answer`
- `fallback`
- `citations[]`

---

## 7. Configuration Summary

Important current runtime configuration includes:

- Milvus host / port / collection / timeouts
- search defaults: `SEARCH_METHOD`, `TOP_K`, `FETCH_K`, `MIN_SCORE`
- hybrid RRF: `HYBRID_RRF_K`
- embedding API config
- rerank provider config
- MongoDB config
- ingest lease seconds
- LLM config

See:

- [.env.example](C:/Users/xieyuxiang/Documents/RAG_medical/.env.example)
- [app/config/settings.py](C:/Users/xieyuxiang/Documents/RAG_medical/app/config/settings.py)

---

## 8. Current Acceptance State

The current implementation already supports:

1. streaming ingest into KB documents
2. idempotent Milvus writes by stable `doc_id`
3. MongoDB-backed resumable ingest
4. asynchronous ingest submission
5. SSE progress updates
6. cooperative cancellation
7. dense / sparse / hybrid retrieval
8. post-fusion rerank
9. fallback-safe answer generation
10. infrastructure-aware early stop for repeated upstream failures

This document should be updated when any of the following changes:

- retrieval contract
- rerank provider contract
- ingest status model
- Mongo persistence model
- Milvus schema
