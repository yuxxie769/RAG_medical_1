# MongoDB 导入断点续跑说明

## 1. 目标

这套导入机制的目标不是“记住上次跑到第几行，然后从下一行继续”，而是：

- 用 MongoDB 作为唯一的导入状态与审计来源
- 允许批次失败后继续处理后续批次
- 在下次普通 `/ingest` 调用时自动补导失败缺口
- 避免 Milvus 已写入、但状态还没落库时造成重复导入或脏状态

当前实现不再依赖本地 `logs/ingest*` JSON 文件，也不再暴露 `resume_from_line` 给调用方。

## 2. 核心概念

### 2.1 source_path + collection_name

系统会先把 `source_path` 规范化为绝对路径，再结合当前目标 `collection_name` 识别“是不是同一个导入目标”。

- 不同绝对路径的 JSONL，会被视为不同数据源
- 相同绝对路径、相同 `collection_name`，会被视为同一个导入生命周期
- 相同绝对路径、不同 `collection_name`，会被视为不同导入生命周期，各自拥有独立的 run、batch 和审计事件

当前版本假设：同一个导入生命周期内，输入 JSONL 不会被原地修改。

### 2.2 ingest_runs

`ingest_runs` 表示一条导入任务主记录，保存：

- `ingest_run_id`
- `source_path`
- `collection_name`
- 请求参数
- 任务状态
- 当前租约
- 汇总统计
- `checkpoint_line`
- `last_scanned_line`

状态含义：

- `running`：当前有请求持有租约并在处理
- `completed`：所有批次都成功，没有失败缺口
- `completed_with_errors`：任务整体跑完，但仍有失败批次
- `failed`：顶层发生不可恢复异常，任务整体失败

### 2.3 ingest_batches

`ingest_batches` 表示按原始行范围切出来的批次，保存：

- `start_line`
- `end_line`
- `status`
- `attempt_count`
- `raw_count`
- `cleaned_count`
- `document_count`
- `indexed_count`
- `existing_count`
- `error`

状态含义：

- `pending`：待处理或可重试
- `running`：当前某个请求正在处理
- `succeeded`：该批次已经成功完成
- `failed`：该批次本次处理失败，等待下次自动补导

### 2.4 ingest_events

`ingest_events` 是追加式审计日志，记录：

- run 创建
- run 恢复
- 批次成功
- 批次失败
- run 完成
- run 异常

它用于追踪过程，不参与主流程判定。

注意：

- 如果批次命中“此前已经成功”，系统会直接跳过该批次
- 这种跳过不会再次写入 `batch_succeeded` 事件
- 重新触发 `/ingest` 时，通常只会新增 run 级事件，例如 `run_resumed` 和新的 `run_finished`
- `GET /ingest/status` 会返回 `execution_outcome`，用于区分本次执行是 `processed` 还是 `skipped_all`

## 3. 普通 `/ingest` 的行为

### 3.1 MongoDB 不可用

系统会先检查 MongoDB 可用性。

- 如果 MongoDB 不可用，`/ingest` 直接返回 `503`
- 此时不会继续写 Milvus

这是为了避免产生无法追踪的向量数据。

### 3.2 同一个 source_path + collection_name 的重复导入

如果你重复导入同一个文件路径，并且目标 `collection_name` 也相同，默认会复用这组 `(source_path, collection_name)` 最近的一条 run，而不是新建一条全新任务。

这时系统会：

- 跳过已经 `succeeded` 的批次
- 自动重试 `failed`、`pending`，以及租约过期后遗留的 `running` 批次
- 最后重新汇总整个 run 的状态

所以重复导入同一个文件时，返回结果经常看起来和上一次很像，这是预期行为。

### 3.3 同一个 source_path 但换了 collection

如果 `source_path` 相同，但 `.env` 中的 `MILVUS_COLLECTION_NAME` 变了：

- 系统会创建新的 run
- 不会复用旧 collection 的 checkpoint、failed batch 或成功批次状态
- 这次导入会被视为新的目标 collection 全量导入

也就是说，当前实现已经按 `source_path + collection_name` 做了状态隔离。

### 3.4 force_reingest=true

如果请求中传：

```json
{
  "source_path": "data/sample.jsonl",
  "force_reingest": true
}
```

系统会为当前 `(source_path, collection_name)` 新建一条新的 `ingest_run`。

注意：

- 它只会新建 run，不会清空 Milvus
- 如果 Milvus 中已有相同稳定 `doc_id`，仍会被视为已存在，只补缺失文档

### 3.5 不同文件的导入

如果导入的是另一个绝对路径的 JSONL：

- 会创建新的 run
- 和旧文件的失败批次、checkpoint、租约完全隔离

## 4. 租约与并发控制

同一个 `(source_path, collection_name)` 在同一时刻只允许一个活跃导入请求。

实现方式：

- run 上保存 `lease_owner`
- run 上保存 `lease_expires_at`
- 请求开始时原子获取租约

行为如下：

- 如果同一个 `(source_path, collection_name)` 已有未过期租约，新的 `/ingest` 返回 `409`
- 如果旧租约已过期，新请求可以接管并恢复任务
- 恢复时，遗留的 `running` 批次会被转成 `pending`，等待自动重试

`force_reingest=true` 也不能绕过同一组 `(source_path, collection_name)` 上的活跃租约。

## 5. 批次处理逻辑

导入时会按 `settings.batch_size` 流式读取原始 JSONL，并组装成一个个行范围批次。

对于每个批次：

1. 先根据 `ingest_run_id + start_line + end_line` 查或建批次记录
2. 如果该批次已经是 `succeeded`，直接跳过
3. 否则把状态标记为 `running`，并增加 `attempt_count`
4. 执行清洗、KB 构建、Milvus 写入
5. 成功后标记为 `succeeded`
6. 失败后标记为 `failed`，记录错误，并继续处理后面的批次

这意味着：

- 单个批次失败不会中断整次扫描
- run 最终可能是 `completed_with_errors`
- 后续再次调用 `/ingest` 会自动补导这些失败批次
- 对于已经 `succeeded` 的批次，不会重复写 `batch_succeeded` 事件，也不会重复写入 Milvus

## 6. checkpoint_line 和 last_scanned_line

这是最容易混淆的两个字段。

### 6.1 checkpoint_line

`checkpoint_line` 表示“连续成功批次”已经推进到哪一行。

它有一个关键约束：

- 绝不会跨过失败缺口

例子：

- 第 1 到 100 行成功
- 第 101 到 200 行失败
- 第 201 到 300 行成功

此时：

- `last_scanned_line = 300`
- `checkpoint_line = 100`

因为第 101 到 200 行存在缺口，所以 checkpoint 不能直接跳到 300。

### 6.2 last_scanned_line

`last_scanned_line` 表示本次扫描过程中，已经读到源文件的哪一行。

它只反映扫描进度，不代表这些行都已经成功导入。

## 7. Milvus 幂等补导

为了处理“Milvus 已经写入，但 MongoDB 状态还没更新”这种中断场景，Milvus 写入做了幂等补缺。

当前逻辑：

- 每个文档使用稳定 `doc_id`
- 写入前先查询哪些 `doc_id` 已存在
- 只插入缺失文档
- 插入后再次校验当前批次全部 `doc_id` 都已存在
- 校验通过，才允许把该批次标记为 `succeeded`

这能覆盖下面这种场景：

1. 批次文档已经写进 Milvus
2. 进程在 MongoDB 更新前退出
3. 下次 `/ingest` 再次处理该批次
4. 系统发现这些 `doc_id` 已存在，只补缺失部分
5. 校验通过后，把批次安全标记为成功

## 8. 重复导入会发生什么

### 8.1 重复导入相同文件到相同 collection

默认是“续跑/补导”语义：

- 复用最近 run
- 已成功批次跳过
- 失败批次重试
- Milvus 已存在的 `doc_id` 不重复写入

如果上一次已经全成功，这次返回通常仍然是：

- 同一个 `ingest_run_id`
- `status=completed`
- 类似的批次统计
- 但事件流中会新增一条 `run_resumed` 和一条新的 `run_finished`
- 不会为已成功批次再写新的 `batch_succeeded`
- 如果这次只是命中历史成功批次、没有实际处理新批次，那么 `execution_outcome=skipped_all`
- 如果这次至少实际处理了一个批次（无论成功还是失败），那么 `execution_outcome=processed`

### 8.2 同文件导入不同 collection

如果 `source_path` 相同，但目标 `collection_name` 不同：

- 会新建 run
- 不会跳过旧 collection 的成功批次
- 不共享 checkpoint、失败缺口和租约

### 8.3 导入不同文件

不同绝对路径会新建 run，不和旧任务共享状态。

### 8.4 同一路径但文件内容变了

当前版本不会自动识别“文件内容已经变化”。

也就是说：

- 如果路径没变，系统仍可能复用旧 run
- 这在语义上不完全安全

更稳妥的做法是：

- 改文件名或路径
- 或者使用 `force_reingest=true`

## 9. 接口说明

### 9.1 POST /ingest

请求示例：

```json
{
  "source_path": "data/sample.jsonl",
  "raw_batch_id": "raw_20260602",
  "clean_batch_id": "clean_20260602",
  "kb_batch_id": "kb_20260602",
  "flush_to_milvus": true,
  "force_reingest": false
}
```

返回重点字段：

- `status`
- `ingest_run_id`
- `succeeded_batches`
- `pending_batches`
- `failed_batches`
- `checkpoint_line`
- `last_scanned_line`

状态说明：

- `completed`：没有失败缺口
- `completed_with_errors`：扫描结束，但仍有失败批次

### 9.2 GET /ingest/status

请求示例：

```text
GET /ingest/status?ingest_run_id=...
```

返回重点字段：

- `status`
- `checkpoint_line`
- `last_scanned_line`
- `failed_batch_details`

`failed_batch_details[]` 包含：

- `batch_id`
- `start_line`
- `end_line`
- `attempt_count`
- `error`

### 9.3 GET /health

健康检查会额外返回 MongoDB 连通状态：

```json
{
  "status": "ok",
  "project": "RAG Medical",
  "mongodb": "ok"
}
```

当 `mongodb` 为 `unavailable` 时，`/ingest` 会返回 `503`。

## 10. 常见场景

### 场景一：中途有一个批次失败

结果：

- 当前请求继续处理后续批次
- run 最终为 `completed_with_errors`
- `checkpoint_line` 停在失败缺口之前
- 下次普通 `/ingest` 自动补导失败批次

### 场景二：服务在 Milvus 写入后崩掉

结果：

- 下次重跑时按稳定 `doc_id` 检查已存在数据
- 只补缺失文档
- 校验完整后把批次标成成功

### 场景三：同一个文件已经导完，又调了一次 `/ingest`

结果：

- 大部分批次直接跳过
- 返回通常和上一次很接近
- 不会因为重复调用产生重复文档
- 不会重复新增 `batch_succeeded`
- 会新增 run 级事件，如 `run_resumed` / `run_finished`

### 场景四：同一个文件切到新的 collection 再导入

结果：

- 会创建新的 run
- 不会复用旧 collection 的成功批次状态
- 会重新扫描并写入新的目标 collection

### 场景五：两个请求同时导同一个文件到同一个 collection

结果：

- 只有一个请求能拿到租约
- 另一个请求返回 `409`

### 场景六：两个请求同时导同一个文件到不同 collection

结果：

- 两边是独立导入目标
- 各自可以创建自己的 run
- 不共享租约

## 11. 使用建议

- 把普通 `/ingest` 当成“导入 + 自动补洞”入口，不需要自己管理续跑行号
- 如果只是想继续之前失败的任务，直接再次调用 `/ingest`
- 如果想明确开启一轮新的 run，使用 `force_reingest=true`
- 如果你切换了 `MILVUS_COLLECTION_NAME`，当前实现会自动按新 collection 建立独立 run
- 如果文件内容变了但路径没变，优先改路径或强制新 run
- 部署时先看 `/health` 里的 `mongodb` 状态，再触发导入

## 12. 租约详解

租约是同一个 `(source_path, collection_name)` 导入任务的限时所有权锁。它解决的问题是：同一个文件导入到同一个目标 collection 时，不能被两个请求同时处理，否则两个进程可能同时处理相同批次、重复写入 Milvus，或者互相覆盖 MongoDB 中的批次状态。

run 中与租约相关的字段：

- `lease_owner`：当前持有租约的执行者标识
- `lease_expires_at`：当前租约自动失效的时间

租约不是永久锁，而是会过期的锁。这样设计是为了覆盖进程强制退出、机器重启或网络中断：这些情况下，旧进程无法主动释放锁，但租约到期后，新请求仍然可以接管任务。

正常流程：

1. `/ingest` 开始时，服务会原子获取租约。
2. 每次开始处理新的批次前，服务会续租。
3. 导入正常完成或遇到顶层异常时，服务会主动释放租约。
4. 如果同一个 `(source_path, collection_name)` 已有未过期租约，新的 `/ingest` 会返回 `409`。

强制退出后的恢复流程：

1. MongoDB 中可能残留 `status=running`、`lease_owner` 和 `lease_expires_at`。
2. 这不代表任务永久卡死，只代表旧进程没有机会主动清理状态。
3. 租约未过期前，相同 `(source_path, collection_name)` 的新 `/ingest` 仍会返回 `409`。
4. 租约过期后，再次调用普通 `/ingest` 会接管原 run。
5. 遗留的 `running` 批次会被转为 `pending`，然后自动重试。

默认租约时长来自：

```text
INGEST_LEASE_SECONDS=300
```

排障时不要只看 run 的 `status=running`。`last_scanned_line` 只表示曾经扫描到哪一行，真正可信的恢复依据是 `ingest_batches.status`。通常不要手工修改 `status`、`checkpoint_line`、`lease_owner` 或 `lease_expires_at`。

## 13. Milvus 连续失败熔断

单个批次失败时，系统仍会记录缺口并继续处理后续批次。但如果连续 3 个批次都发生 Milvus 写入阶段故障，继续扫描通常没有收益，因此系统会触发熔断：

```text
Milvus 整体不可用
→ 连续 3 个 Milvus 写入批次失败
→ 第 3 个失败批次先完整记录到 MongoDB
→ 写入 circuit_breaker_opened 审计事件
→ 当前 run 标记为 failed 并释放租约
→ 停止继续扫描后续 JSONL
```

Milvus 恢复后，再次调用相同 `(source_path, collection_name)` 的普通 `/ingest` 即可恢复：

- 复用原 run
- 跳过已经成功的批次
- 自动重试失败批次
- 根据稳定 `doc_id` 跳过 Milvus 中已经存在的文档

熔断只针对 Milvus 写入阶段故障。清洗、KB 构建或 embedding 异常仍按普通失败批次处理，不会误触发 Milvus 熔断。

## 14. 异步导入、SSE、取消与逐批日志

当前 `POST /ingest` 已经改为异步提交。

### 14.1 POST /ingest

请求进入后只做这几件事：

1. 校验 `source_path`
2. 检查 MongoDB
3. 获取或恢复 run 租约
4. 创建后台任务
5. 立即返回 `202 Accepted`

返回示例：

```json
{
  "status": "queued",
  "ingest_run_id": "run_xxx",
  "source_path": "C:\\data\\sample.jsonl",
  "status_url": "/ingest/status?ingest_run_id=run_xxx",
  "events_url": "/ingest/run_xxx/events",
  "cancel_url": "/ingest/run_xxx"
}
```

也就是说，`POST /ingest` 不再等待整次导入跑完才返回。

### 14.2 GET /ingest/status

状态接口除了原有的批次和 checkpoint 信息外，还会返回本轮执行指标：

- `stage`
- `cancel_requested`
- `total_lines`
- `current_scanned_line`
- `progress_percent`
- `processed_batches`
- `skipped_succeeded_batches`
- `execution_outcome`
- `elapsed_seconds`
- `lines_per_second`
- `batches_per_second`
- `embedding_seconds_total`
- `embedding_seconds_last_batch`
- `eta_seconds`

`stage` 的含义：

- `queued`：任务已创建，等待后台执行
- `counting_lines`：正在统计总行数
- `ingesting`：正在处理批次
- `cancelling`：已收到取消请求，等待当前批次结束
- `finished`：任务已进入终态

### 14.3 SSE 进度流

新增接口：

```text
GET /ingest/{ingest_run_id}/events
```

它返回 `text/event-stream`，用途是实时看进度，不用反复手工轮询。

行为如下：

- 连接建立后立即推送一次当前快照
- 后台状态变化时继续推送 `progress` 事件
- 15 秒没有变化时发送 heartbeat
- 任务进入 `cancelled`、`completed`、`completed_with_errors` 或 `failed` 后，推送最后一次快照并关闭连接

### 14.4 取消接口

新增接口：

```text
DELETE /ingest/{ingest_run_id}
```

取消是协作式取消，不会粗暴中断当前正在执行的 embedding 或 Milvus RPC。

收到取消请求后：

- run 先变成 `cancelling`
- 当前批次处理结束后停止继续扫描
- 最终状态写成 `cancelled`
- 租约会被释放

### 14.5 逐批日志

现在每个原始批次结束都会记一条日志，不管结果是：

- `batch_succeeded`
- `batch_failed`
- `batch_skipped`

日志会同时写到两处：

- 应用标准日志
- `ingest_events`

每条批次日志都会带上：

- `ingest_run_id`
- `batch_id`
- `start_line`
- `end_line`
- `attempt_count`
- `raw_count`
- `cleaned_count`
- `document_count`
- `indexed_count`
- `existing_count`
- `batch_seconds`
- `embedding_seconds`
- `processed_batches`
- `lines_per_second`
- `batches_per_second`
- `eta_seconds`
- `error`

日志不会写入问答正文，只记录定位和排障需要的运行信息。

## 15. 2026-06 中断恢复与租约状态补充

这一轮改动主要解决的是“worker 中途异常退出后，run 长时间停留在 `running` / `cancelling`，管理台和调用方很难判断任务到底还活不活”的问题。

### 15.1 新增 `interrupted` 状态

`interrupted` 表示：

- 上一次执行没有正常收尾
- 当前没有有效执行者继续持有这条 run
- 这条任务现在可以重新恢复，也可以直接取消

它和 `failed` 的区别是：

- `failed`：本轮执行遇到了顶层不可恢复异常，任务已经明确失败
- `interrupted`：更偏向“执行者失联或服务重启后的遗留运行态”，恢复语义仍然成立

### 15.2 启动时自动收敛遗留运行态

应用启动时，会扫描遗留的：

- `queued`
- `running`
- `cancelling`

并且仍然带有 `lease_owner` 的 run，把它们统一收敛成 `interrupted`。

如果日志里看到类似：

```text
Reconciled 2 orphaned ingest runs into interrupted state
```

意思是：

- 本次启动自动发现了 2 条遗留运行态任务
- 它们已经被系统改写成 `interrupted`
- 这不是报错，而是恢复机制正常生效

### 15.3 读取状态时对 stale / abandoned run 的解释

即使服务还没重启，只要 `GET /ingest/status` 发现：

- 旧状态仍然是 `running` 或 `cancelling`
- 但 `lease_expires_at` 已经过期

就会把这条 run 按 `interrupted` 语义返回，并额外带上两个辅助标记：

- `stale=true`：说明旧租约已经过期，当前状态属于遗留运行态解释结果
- `abandoned=true`：说明这条任务很可能是在处理中途失联的

这两个字段主要给管理台、排障脚本和运维接口调用方使用，用来识别“看起来像还在跑、实际上已经没有活 worker”的任务。

### 15.4 对恢复行为的影响

这次改动没有改变原来的批次级断点恢复原则：

- 已成功的 batch 仍然会被跳过
- `failed` / `pending` 的 batch 仍然会在下一次普通 `/ingest` 时自动补导
- 同一 `(source_path, collection_name)` 的恢复仍然以 MongoDB 中的 `ingest_batches` 为准

区别在于：

- 以前更依赖“等 lease 自然过期”
- 现在服务重启后会主动把遗留运行态收敛成 `interrupted`
- 管理台和调用方能更快看出任务已经中断，而不是继续误判成 `running`

### 15.5 对取消行为的影响

取消接口现在分两种情况：

1. 普通运行中任务  
   仍然保持协作式取消：
   - run 先进入 `cancelling`
   - 等当前批次结束后再进入 `cancelled`

2. 已经 `interrupted` 且没有 `lease_owner` 的任务  
   可以直接收尾成 `cancelled`，不会再长时间挂在 `cancelling`

这样做的目的，是避免“worker 已死，但 run 永远只停在 `cancelling`”这种半死不活的状态。

### 15.6 现在怎么看一条任务是否还真的活着

排障时不要只看：

- `status=running`
- `last_scanned_line`

还要一起看：

- 是否还有有效 `lease_owner`
- `lease_expires_at` 是否已经过期
- `GET /ingest/status` 返回里是否出现 `stale=true` / `abandoned=true`
- 批次层面的 `ingest_batches.status`

在当前版本里：

- `running` / `cancelling` 更适合解释为“存在活跃执行者或尚未被收敛”
- `interrupted` 明确表示“当前没有活跃执行者，但可恢复”
- 真正可信的恢复依据仍然是批次状态，而不是单个 run 级汇总字段

## 16. 熔断补充说明

当前实际生效的熔断规则比上面的旧描述更简单：

- 连续 3 个批次发生基础设施故障，就停止当前导入
- 基础设施故障只分两类：`embedding` 和 `milvus`
- 其他错误，例如清洗异常、KB 构建异常、单条脏数据问题，仍然只记失败批次，不触发提前停止

这里的 `embedding` 包括：

- API 配额耗尽
- 429
- embedding 鉴权失败
- 上游 embedding 服务不可用

这里的 `milvus` 包括：

- collection 检查或创建失败
- 连接断开
- 写入失败
- 持久化校验失败

熔断触发流程：

```text
连续 3 个批次出现 embedding 或 Milvus 基础设施故障
→ 第 3 个失败批次先完整写入 MongoDB
→ 写入 circuit_breaker_opened 审计事件
→ 当前 run 标记为 failed 并释放租约
→ 停止继续扫描后续 JSONL
```

修复基础设施后，再次调用同一个 `(source_path, collection_name)` 的普通 `/ingest` 即可恢复：

- 复用原 run
- 跳过成功批次
- 自动重试失败批次
- 继续依赖稳定 `doc_id` 做 Milvus 幂等补导
