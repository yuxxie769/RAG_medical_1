# RAG Medical

医疗知识问答项目，目标是基于 RAG 能力提供中文健康与疾病相关问题的辅助问答。

当前仓库已经完成后端主链路，并进入第二阶段的前后端功能完善阶段。`README` 主要用于帮助你快速理解项目、完成本地启动和找到关键入口；更细的技术设计放在 `docs/` 中维护。

## 当前阶段

当前处于第二阶段：

- 后端核心链路已可用，包括数据导入、检索召回、回答生成和导入状态跟踪。
- 第二阶段重点是补齐前端交互界面、管理入口和页面层交互逻辑。
- 当前仓库仍以后端能力为主，前端架构和任务清单已经整理到 `docs/`。

## 当前能力

- 支持从 JSONL 文件导入医疗问答数据。
- 支持数据清洗、知识库构建和向量入库。
- 支持 `hybrid`、`dense`、`sparse` 三种检索模式。
- 支持基于检索结果生成最终回答。
- 支持无结果时的 fallback 安全降级。
- 支持 MongoDB 记录导入任务状态、失败批次和进度事件。
- 支持异步导入、状态查询、SSE 实时进度和取消导入。

## 项目结构

- `app/`：核心业务代码
- `tests/`：测试代码
- `scripts/`：辅助脚本
- `docs/`：架构、流程和阶段文档

## 快速启动

### 1. 安装依赖

```bash
uv sync
```

### 2. 准备配置

以 `.env.example` 为参考，在项目根目录准备 `.env`。

最少需要确认这些配置：

- Milvus 连接信息
- MongoDB 连接信息
- Embedding API Key
- LLM API Key

如果暂时不启用 rerank，可以保持默认关闭。

### 3. 启动服务

```bash
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后可访问：

- 健康检查：`http://127.0.0.1:8000/health`
- Swagger 文档：`http://127.0.0.1:8000/docs`

## 配置项说明

`README` 只保留配置分组和用途说明，完整字段见 [.env.example](C:/Users/xieyuxiang/Documents/RAG_medical/.env.example)。

### 1. 项目与默认检索参数

这组配置决定项目名称、默认检索行为和超时：

- `PROJECT_NAME`
- `ENVIRONMENT`
- `TOP_K`
- `FETCH_K`
- `MIN_SCORE`
- `SEARCH_METHOD`
- `RETRIEVAL_TIMEOUT_SECONDS`
- `GENERATION_TIMEOUT_SECONDS`

适用场景：

- 想统一默认召回数量和返回数量
- 想调整接口超时阈值
- 想切换默认检索模式

### 2. Milvus 配置

这组配置用于连接向量库和指定目标 collection：

- `MILVUS_HOST`
- `MILVUS_PORT`
- `MILVUS_COLLECTION_NAME`
- `MILVUS_REQUEST_TIMEOUT_SECONDS`
- `MILVUS_MANAGEMENT_TIMEOUT_SECONDS`
- `ENABLE_MILVUS`

通常只需要先关注：

- 地址是否正确
- collection 名称是否符合当前环境
- 本地或测试环境是否真的启用了 Milvus

### 3. Embedding 配置

这组配置决定向量化服务从哪里调用：

- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_DIMENSION`
- `EMBEDDING_TIMEOUT_SECONDS`

关注重点：

- API Key 是否可用
- 模型和维度是否与当前 Milvus collection 兼容

### 4. Rerank 配置

这组配置控制是否启用重排序能力：

- `ENABLE_RERANK`
- `RERANK_PROVIDER`
- `RERANK_API_KEY`
- `RERANK_BASE_URL`
- `RERANK_PATH`
- `RERANK_MODEL_NAME`
- `RERANK_TIMEOUT_SECONDS`
- `RERANK_CANDIDATE_LIMIT`

说明：

- 默认可以不开启。
- 只有在你已经准备好 rerank 服务时，才需要补齐整组配置。

### 5. MongoDB 与导入状态配置

这组配置用于导入任务状态、断点续跑和审计记录：

- `MONGODB_URI`
- `MONGODB_DATABASE`
- `MONGODB_CONNECT_TIMEOUT_MS`
- `INGEST_LEASE_SECONDS`

说明：

- MongoDB 不可用时，导入接口会拒绝开始任务。
- 如果你要验证导入流程，这组配置必须先可用。

### 6. LLM 配置

这组配置用于最终回答生成：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL_NAME`
- `LLM_TIMEOUT_SECONDS`

### 7. 批处理配置

这组配置主要影响导入批次大小：

- `BATCH_SIZE`

通常不需要一开始就调整，除非你在做性能测试或处理非常大的数据文件。

## 常用接口

### 健康检查

- `GET /health`

### 数据导入

- `POST /ingest`
- `GET /ingest/status?ingest_run_id=...`
- `GET /ingest/{ingest_run_id}/events`
- `DELETE /ingest/{ingest_run_id}`

### 检索与回答

- `POST /query`
- `POST /answer`

如果只是想验证链路是否通，通常顺序是：

1. 先检查 `/health`
2. 再执行 `/ingest`
3. 导入完成后调用 `/query` 或 `/answer`

## 使用说明

### 导入数据

当前导入接口为异步模式。提交导入后，接口会先返回一个 `ingest_run_id`，然后通过状态查询或 SSE 继续跟踪进度。

适合配合以下方式查看：

- 轮询 `GET /ingest/status`
- 订阅 `GET /ingest/{ingest_run_id}/events`

### 检索与回答

`/query` 用于查看检索命中，`/answer` 用于直接获取最终回答。

如果当前没有可用索引数据，相关接口不会返回正常结果，因此第一次使用前需要先完成导入。

## 文档索引

如果你想看更细的实现说明，直接看这些文档：

- [后端架构](C:/Users/xieyuxiang/Documents/RAG_medical/docs/architecture.md)
- [数据流与幂等设计](C:/Users/xieyuxiang/Documents/RAG_medical/docs/dataflow_overview.md)
- [导入断点续跑说明](C:/Users/xieyuxiang/Documents/RAG_medical/docs/ingest_resume.md)
- [前端架构](C:/Users/xieyuxiang/Documents/RAG_medical/docs/前端架构.md)
- [前端第二阶段流程任务核查清单](C:/Users/xieyuxiang/Documents/RAG_medical/docs/前端第二阶段流程任务核查清单.md)

## 后续重点

当前更适合继续推进的方向：

- 页面层问答入口与管理入口落地
- 引用展示和会话管理
- 导入管理页与 SSE 实时进度展示
- 评测与部署阶段准备
