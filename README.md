# RAG Medical

医疗知识问答 RAG 后端项目第一阶段骨架。

## 当前阶段范围
- 数据读取与清洗
- 知识库构建
- 文档向量化
- 检索召回
- 融合召回（dense + sparse + RRF）
- LLM 回答生成
- fallback 安全降级
- 日志记录与基础评估

## 已完成
- 项目目录骨架
- 核心数据模块骨架
- 数据模块支持流式读取、行号定位和批次信息
- 知识库构建模块骨架
- 检索、向量化、Milvus 原生融合召回
- 真实 Milvus SDK 接入
- LLM 生成模块接入
- 一键导入、检索、答案生成 API

## 当前开发约定
### 数据与批次
- `raw_batch_id`：原始读取批次
- `clean_batch_id`：清洗批次
- `kb_batch_id`：知识库构建批次
- `doc_id`：由知识库层自动生成，采用稳定哈希

### 检索与入库
- 数据主流程采用流式读取，不一次性全量载入
- 向量入库采用分批处理
- 检索链路支持 `hybrid`、`dense`、`sparse` 三种模式，默认使用 `hybrid`
- Milvus 是第一阶段目标向量数据库
- 重复 `doc_id` 的批次会报错并跳过，不会污染库

### embedding 接入
当前支持 Qwen Cloud OpenAI-compatible embedding：
- base_url：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- model：`text-embedding-v4`
- dimensions：`1024`
- encoding_format：`float`

### LLM 接入
当前支持 Qwen Cloud OpenAI-compatible chat completion：
- base_url：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- model：`qwen-plus`
- timeout：`30`

## 启动方式

本项目的 FastAPI 启动入口在 `app/api/main.py`，可以使用 `uv` 启动服务：

```bash
uv sync
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后可以访问：
- 健康检查：`http://127.0.0.1:8000/health`
- Swagger 文档：`http://127.0.0.1:8000/docs`

## 如何测试数据与检索骨架

可以先运行：
```python
from app.retrieval import DummyEmbedder, InMemoryMilvusStore, MilvusCollectionSchema, VectorIndexer
```

真实接入时使用：
```python
from app.retrieval import OpenAICompatibleEmbedder
```

## 目录结构
- `app/`：核心业务代码
- `scripts/`：批处理脚本
- `tests/`：测试代码
- `docs/`：设计文档

## Milvus 混合召回

`POST /query` 默认使用 Milvus 原生混合召回，也可以通过 `search_method` 指定单路检索：

```json
{
  "query": "口干怎么办？",
  "search_method": "hybrid",
  "top_k": 5,
  "fetch_k": 15,
  "min_score": 0.0
}
```

`search_method` 支持 `hybrid`、`dense` 和 `sparse`。其中 dense 使用 `COSINE`，sparse 使用 Milvus BM25 Function，hybrid 使用 `RRFRanker(k=60)`。

## Rerank 阶段

当前检索链路支持一个可插拔的 rerank 阶段，位置在融合召回之后、最终截断 `top_k` 之前。

默认关闭，可通过环境变量开启：

```text
ENABLE_RERANK=true
RERANK_PROVIDER=openai_compatible
RERANK_API_KEY=...
RERANK_BASE_URL=https://your-endpoint.example.com
RERANK_PATH=/rerank
RERANK_MODEL_NAME=...
RERANK_TIMEOUT_SECONDS=10
RERANK_CANDIDATE_LIMIT=15
```

当前内置两个 provider：

- `none`
- `openai_compatible`

rerank 会对 `dense`、`sparse`、`hybrid` 三种检索模式统一生效。  
如果 rerank 服务超时、429、额度不足或返回异常，请求会自动降级回原始召回顺序，不会让 `/query` 或 `/answer` 失败。

响应中的每个命中会在 `metadata` 中补充：

- `retrieval_score`
- `rerank_score`
- `rerank_applied`
- `rerank_provider`

重建 collection 时运行：

```bash
python scripts/reset_milvus_collection.py --confirm-drop
```

重建后调用 `POST /ingest` 并传入 `"force_reingest": true`，以忽略旧断点并重新导入数据。

## 下一步
继续补充生成安全策略和完整评估流程。

## MongoDB 导入状态与断点续跑

导入状态和审计事件已迁移到 MongoDB，运行时不再读取或写入 `logs/ingest*` 文件。

- `ingest_runs`：保存导入任务、租约、汇总统计和连续成功断点。
- `ingest_batches`：保存行区间级批次状态。失败批次会保留缺口，不阻断后续批次。
- `ingest_events`：保存任务和批次审计事件。

再次调用 `POST /ingest` 时，服务会自动跳过成功批次并重试失败批次。`force_reingest=true`
会创建新的导入任务。MongoDB 不可用时导入接口返回 `503`，避免产生无法追踪的 Milvus 写入。

## 导入断点续跑文档

完整说明见 [docs/ingest_resume.md](C:/Users/xieyuxiang/Documents/RAG_medical/docs/ingest_resume.md)。

当前 `POST /ingest` 已改为异步 `202 Accepted`。导入创建后可用：

- `GET /ingest/status?ingest_run_id=...` 查看状态与速率指标
- `GET /ingest/{ingest_run_id}/events` 通过 SSE 订阅实时进度
- `DELETE /ingest/{ingest_run_id}` 请求取消当前导入
