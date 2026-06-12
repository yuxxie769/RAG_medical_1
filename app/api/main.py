from __future__ import annotations

import json
import time
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.admin import ADMIN_COOKIE_NAME, AdminAuthUnavailable, AdminRepository, AdminSessionError, read_admin_session_value
from app.chat import ChatRepository, ChatService
from app.config.settings import settings
from app.core.logging import get_logger
from app.core.models import RetrievalResult
from app.generation import Generator, normalize_answer_citations
from app.ingest import IngestLeaseConflict, IngestService, MongoIngestRepository, MongoUnavailable
from app.ingest.repository import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES
from app.retrieval import (
    DummyEmbedder,
    HybridRetriever,
    MilvusCollectionSchema,
    MilvusOperationError,
    MilvusStore,
    NoopReranker,
    LocalHTTPEmbedder,
    OpenAICompatibleEmbedder,
    OpenAICompatibleReranker,
    VectorIndexer,
)


app = FastAPI(title=settings.project_name)
logger = get_logger(__name__)

# 全局数据存储
_DATASTORE = {
    "embedder": None,
    "store": None,
    "generator": None,
    "retriever": None,
    "ingest_repository": None,
    "ingest_service": None,
    "admin_repository": None,
    "chat_repository": None,
    "chat_service": None,
}


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to raw jsonl file")
    raw_batch_id: str | None = Field(default=None, description="Optional raw batch id")
    clean_batch_id: str | None = Field(default=None, description="Optional clean batch id")
    kb_batch_id: str | None = Field(default=None, description="Optional knowledge base batch id")
    flush_to_milvus: bool = Field(default=True, description="Whether to write into Milvus after indexing")
    force_reingest: bool = Field(default=False, description="Create a new ingest run and rescan the source")

#run任务失败详情
class FailedBatchDetail(BaseModel):
    batch_id: str
    start_line: int
    end_line: int
    attempt_count: int = 0
    error: str | None = None

#run任务接受响应
class IngestAcceptedResponse(BaseModel):
    status: Literal["queued"]
    ingest_run_id: str
    source_path: str
    status_url: str
    events_url: str
    cancel_url: str


#run任务状态响应
class IngestStatusResponse(BaseModel):
    ingest_run_id: str
    exists: bool
    status: str | None = None
    stage: str | None = None
    stale: bool = False
    abandoned: bool = False
    execution_outcome: Literal["not_started", "processed", "skipped_all"] | None = None
    source_path: str | None = None
    raw_count: int = 0
    cleaned_count: int = 0
    document_count: int = 0
    indexed_count: int = 0
    succeeded_batches: int = 0
    pending_batches: int = 0
    failed_batches: int = 0
    checkpoint_line: int = 0
    last_scanned_line: int = 0
    current_scanned_line: int = 0
    cancel_requested: bool = False
    total_lines: int = 0
    progress_percent: float = 0.0
    processed_batches: int = 0
    skipped_succeeded_batches: int = 0
    elapsed_seconds: float | None = None
    lines_per_second: float | None = None
    batches_per_second: float | None = None
    embedding_seconds_total: float = 0.0
    embedding_seconds_last_batch: float = 0.0
    eta_seconds: float | None = None
    failed_batch_details: list[FailedBatchDetail] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    search_method: Literal["hybrid", "dense", "sparse"] = Field(default=settings.search_method)
    top_k: int = Field(default=settings.top_k, ge=1, le=100)
    fetch_k: int | None = Field(default=None, ge=1, le=500)
    min_score: float | None = Field(default=None)


class QueryHit(BaseModel):
    doc_id: str
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)

#查询响应
class QueryResponse(BaseModel):
    query: str
    search_method: Literal["hybrid", "dense", "sparse"]
    top_k: int
    fetch_k: int
    min_score: float
    hits: list[QueryHit] = Field(default_factory=list)

#回答请求
class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=100)

#回答响应
class AnswerResponse(BaseModel):
    query: str
    answer: str
    fallback: bool = False
    citations: list[QueryHit] = Field(default_factory=list)
    retrieval_results: list[QueryHit] = Field(default_factory=list)

# 初始化embbder使用emb api key构建emb对象
def _build_embedder():
    embedding_base_url = settings.embedding_base_url.strip()
    if embedding_base_url.rstrip("/").endswith("/embeddings"):
        logger.info("Using local HTTP embedder endpoint=%s", embedding_base_url)
        return LocalHTTPEmbedder(
            endpoint=embedding_base_url,
            timeout=settings.embedding_timeout_seconds,
            expected_dimension=settings.embedding_dimension,
        )
    if settings.embedding_api_key:
        logger.info("Using OpenAI-compatible embedder with remote API.")
        return OpenAICompatibleEmbedder(
            api_key=settings.embedding_api_key,
            base_url=embedding_base_url,
            model=settings.embedding_model_name,
            dimensions=settings.embedding_dimension,
            timeout=settings.embedding_timeout_seconds,
        )
    logger.info("Using dummy embedder for local development.")
    return DummyEmbedder(settings.embedding_dimension)

# 初始化store使用milvus_collection_name构建store对象
def _build_store() -> MilvusStore:
    schema = MilvusCollectionSchema(
        collection_name=settings.milvus_collection_name,
        dimension=settings.embedding_dimension,
        dense_metric_type=settings.milvus_dense_metric_type,
        sparse_vector_field_name=settings.milvus_sparse_field_name,
        analyzer_type=settings.milvus_analyzer_type,
        rrf_k=settings.hybrid_rrf_k,
    )
    return MilvusStore(
        schema=schema,
        host=settings.milvus_host,
        port=settings.milvus_port,
        request_timeout_seconds=settings.milvus_request_timeout_seconds,
        management_timeout_seconds=settings.milvus_management_timeout_seconds,
    )

# 合并初始化indexer对象
def _new_indexer() -> VectorIndexer:
    return VectorIndexer(
        embedder=_build_embedder(),
        store=_build_store(),
        batch_size=settings.batch_size,
    )


def _get_store() -> MilvusStore:
    if _DATASTORE["store"] is None:
        _DATASTORE["store"] = _build_store()
    return _DATASTORE["store"]


def _build_reranker():
    if not settings.enable_rerank:
        return NoopReranker()
    if settings.rerank_provider == "none":
        return NoopReranker()
    if settings.rerank_provider == "openai_compatible":
        return OpenAICompatibleReranker(
            api_key=settings.rerank_api_key,
            base_url=settings.rerank_base_url,
            path=settings.rerank_path,
            model=settings.rerank_model_name,
            timeout=settings.rerank_timeout_seconds,
        )
    raise ValueError(f"Unsupported rerank provider: {settings.rerank_provider}")

# retriver包含embedder、store、reranker
def _get_retriever() -> HybridRetriever:
    if _DATASTORE["retriever"] is None:
        if _DATASTORE["embedder"] is None:
            _DATASTORE["embedder"] = _build_embedder()
        _DATASTORE["retriever"] = HybridRetriever(
            embedder=_DATASTORE["embedder"],
            store=_get_store(),
            reranker=_build_reranker(),
            rerank_candidate_limit=settings.rerank_candidate_limit,
        )
    return _DATASTORE["retriever"]


def _get_generator() -> Generator:
    if _DATASTORE["generator"] is None:
        _DATASTORE["generator"] = Generator(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model_name,
            timeout=settings.llm_timeout_seconds,
        )
    return _DATASTORE["generator"]

# 获取ingest数据层对象
def _get_ingest_repository() -> MongoIngestRepository:
    if _DATASTORE["ingest_repository"] is None:
        _DATASTORE["ingest_repository"] = MongoIngestRepository(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
            connect_timeout_ms=settings.mongodb_connect_timeout_ms,
            lease_seconds=settings.ingest_lease_seconds,
        )
    return _DATASTORE["ingest_repository"]

# 获取ingest的service对象，包含数据层对象、indexer、batch size，
def _get_ingest_service() -> IngestService:
    if _DATASTORE["ingest_service"] is None:
        _DATASTORE["ingest_service"] = IngestService(
            repository=_get_ingest_repository(),
            indexer_factory=_new_indexer,
            batch_size=settings.batch_size,
        )
    return _DATASTORE["ingest_service"]

# 为admin数据层搭建一个TCP 连接池
def _get_admin_repository() -> AdminRepository:
    if _DATASTORE["admin_repository"] is None:
        _DATASTORE["admin_repository"] = AdminRepository(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
            connect_timeout_ms=settings.mongodb_connect_timeout_ms,
        )
    return _DATASTORE["admin_repository"]


def _get_chat_repository() -> ChatRepository:
    if _DATASTORE["chat_repository"] is None:
        _DATASTORE["chat_repository"] = ChatRepository(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
            connect_timeout_ms=settings.mongodb_connect_timeout_ms,
        )
    return _DATASTORE["chat_repository"]


def _get_chat_service() -> ChatService:
    if _DATASTORE["chat_service"] is None:
        _DATASTORE["chat_service"] = ChatService(repository=_get_chat_repository())
    return _DATASTORE["chat_service"]

# 获取admin数据层，初始化admin账号
def _ensure_bootstrap_admin() -> None:
    if not settings.admin_bootstrap_username.strip() or not settings.admin_bootstrap_password:
        return
    try:
        _get_admin_repository().ensure_bootstrap_admin(
            settings.admin_bootstrap_username,
            settings.admin_bootstrap_password,
        )
        logger.info("Ensured bootstrap admin user=%s", settings.admin_bootstrap_username.strip())
    except AdminAuthUnavailable as exc:
        logger.warning("Skipping bootstrap admin initialization: %s", exc)

# 应用启动时清理旧进程残留的活跃 run，防止前端看到永远 "running" 的假象。
def _reconcile_interrupted_runs() -> None:
    try:
        repository = _get_ingest_repository()
        reconcile = getattr(repository, "reconcile_interrupted_runs", None) #检查数据层对象有无reconcile_interrupted_runs这个方法
        if reconcile is None:
            return
        reconciled = reconcile() #执行reconcile_interrupted_runs方法
        if reconciled:
            logger.warning("Reconciled %s orphaned ingest runs into interrupted state", reconciled)
    except MongoUnavailable as exc:
        logger.warning("Skipping ingest run reconciliation: %s", exc)

# admin 认证
def _authenticate_admin_cookie(cookie_value: str | None) -> dict:
    try:
        username_normalized = read_admin_session_value(settings.admin_session_secret, cookie_value) #利用私钥admin_session_secret，检查用户的cookie签名
        admin_user = _get_admin_repository().get_active_user(username_normalized) # 拿解析出的用户名去 admin_users 集合里查这个人还在不在、是否 is_active=True
    except AdminSessionError as exc:
        raise PermissionError(str(exc)) from exc
    except AdminAuthUnavailable:
        raise
    if admin_user is None:
        raise PermissionError("Admin authentication required.")
    return admin_user

# admin 认证包装
def _require_admin_api(request: Request) -> dict:
    try:
        return _authenticate_admin_cookie(request.cookies.get(ADMIN_COOKIE_NAME))
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail="Admin authentication required.") from exc
    except AdminAuthUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

# 检查milvus是否有数据
def _ensure_indexed_store() -> MilvusStore:
    store = _get_store()
    try:
        count = store.count()
    except MilvusOperationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Milvus store unavailable: {exc}") from exc
    if count <= 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Call /ingest first.")
    return store

# ingest response格式转换组装函数（dict-》pydantic）
def _to_ingest_status_response(state: dict) -> IngestStatusResponse:
    return IngestStatusResponse(
        ingest_run_id=state["ingest_run_id"],
        exists=True,
        status=state.get("status"),
        stage=state.get("stage"),
        stale=bool(state.get("stale", False)),
        abandoned=bool(state.get("abandoned", False)),
        execution_outcome=state.get("execution_outcome"),
        source_path=state.get("source_path"),
        raw_count=int(state.get("raw_count", 0)),
        cleaned_count=int(state.get("cleaned_count", 0)),
        document_count=int(state.get("document_count", 0)),
        indexed_count=int(state.get("indexed_count", 0)),
        succeeded_batches=int(state.get("succeeded_batches", 0)),
        pending_batches=int(state.get("pending_batches", 0)),
        failed_batches=int(state.get("failed_batches", 0)),
        checkpoint_line=int(state.get("checkpoint_line", 0)),
        last_scanned_line=int(state.get("last_scanned_line", 0)),
        current_scanned_line=int(state.get("current_scanned_line", 0)),
        cancel_requested=bool(state.get("cancel_requested", False)),
        total_lines=int(state.get("total_lines", 0)),
        progress_percent=float(state.get("progress_percent", 0.0)),
        processed_batches=int(state.get("processed_batches", 0)),
        skipped_succeeded_batches=int(state.get("skipped_succeeded_batches", 0)),
        elapsed_seconds=state.get("elapsed_seconds"),
        lines_per_second=state.get("lines_per_second"),
        batches_per_second=state.get("batches_per_second"),
        embedding_seconds_total=float(state.get("embedding_seconds_total", 0.0)),
        embedding_seconds_last_batch=float(state.get("embedding_seconds_last_batch", 0.0)),
        eta_seconds=state.get("eta_seconds"),
        failed_batch_details=state.get("failed_batch_details", []),
    )


# run ingest 方法
def _run_ingest_background(prepared_run) -> None:
    try:
        _get_ingest_service().execute_prepared_ingest(prepared_run)
    except Exception:
        logger.exception("Background ingest execution failed run_id=%s", prepared_run.ingest_run_id)

#格式化字典为json，并适配sse格式
def _format_sse(event: str, data: dict | None = None) -> str:
    if data is None:
        return f"event: {event}\n\n"
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


#########################################路由部分###############################################

# 服务启动后，自动初始化管理员账号 + 修复遗留异常任务
@app.on_event("startup")
def bootstrap_admin_user() -> None:
    _ensure_bootstrap_admin()
    _reconcile_interrupted_runs()


@app.get("/health")
def health_check() -> dict:
    try:
        _get_ingest_repository().ping() # ① 检查 MongoDB TCP 连通性
        mongodb_status = "ok"
    except MongoUnavailable:
        mongodb_status = "unavailable"
    return {"status": "ok", "project": settings.project_name, "mongodb": mongodb_status}


# ingest路由，输入IngestRequest，输出IngestAcceptedResponse
@app.post("/ingest", response_model=IngestAcceptedResponse, status_code=202)
def ingest_data(
    request: IngestRequest,
    background_tasks: BackgroundTasks, #FastAPI 内置后台任务实例，用于注册为后台任务
    _: dict = Depends(_require_admin_api), # 依赖注入：必须先认证
) -> IngestAcceptedResponse:
    try:
        # 准备ingest任务，校验路径 + 找/建 run + 拿租约
        prepared_run = _get_ingest_service().prepare_ingest(
            **request.model_dump(), #注入请求参数
            target_collection_name=settings.milvus_collection_name, # 额外注入目标collection名
        )
        #把 prepare_ingest 返回的 PreparedIngestRun 作为参数，_run_ingest_background作为运行体，后台添加运行任务
        background_tasks.add_task(_run_ingest_background, prepared_run) 
        return IngestAcceptedResponse(
            status="queued",
            ingest_run_id=prepared_run.ingest_run_id,
            source_path=prepared_run.source_path,
            status_url=f"/ingest/status?ingest_run_id={prepared_run.ingest_run_id}",
            events_url=f"/ingest/{prepared_run.ingest_run_id}/events",
            cancel_url=f"/ingest/{prepared_run.ingest_run_id}",
        )
    except MongoUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except IngestLeaseConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# 获取run任务状态
@app.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status(ingest_run_id: str, _: dict = Depends(_require_admin_api)) -> IngestStatusResponse:
    try:
        state = _get_ingest_repository().get_status(ingest_run_id)
    except MongoUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state is None:
        return IngestStatusResponse(ingest_run_id=ingest_run_id, exists=False)
    return _to_ingest_status_response(state)


@app.delete("/ingest/{ingest_run_id}", response_model=IngestStatusResponse)
def cancel_ingest(ingest_run_id: str, _: dict = Depends(_require_admin_api)) -> IngestStatusResponse:
    try:
        state = _get_ingest_repository().request_cancel(ingest_run_id)
    except MongoUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail=f"Ingest run not found: {ingest_run_id}")
    return _to_ingest_status_response(state)

# SSE 实时进度流接口，输出StreamingResponse
@app.get("/ingest/{ingest_run_id}/events")
def ingest_events(ingest_run_id: str, _: dict = Depends(_require_admin_api)) -> StreamingResponse:
    try:
        initial_state = _get_ingest_repository().get_status(ingest_run_id) # 先查一次 run 是否存在与数据库。不存在 → 404，MongoDB 挂了 → 503。
    except MongoUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if initial_state is None:
        raise HTTPException(status_code=404, detail=f"Ingest run not found: {ingest_run_id}")

    # 每隔 1 秒轮询 MongoDB（ingest数据层），比较 updated_at 是否有变化。
    def event_stream():
        last_updated_at = None
        last_heartbeat = time.monotonic() #开始计时
        while True:
            # 再查 run 当前状态
            try:
                state = _get_ingest_repository().get_status(ingest_run_id) 
            
            # 分支1：MongoDB 挂了
            except MongoUnavailable as exc:
                payload = {
                    "ingest_run_id": ingest_run_id,
                    "exists": True,
                    "status": "failed",
                    "failed_batch_details": [],
                    "stage": "finished",
                    "source_path": initial_state.get("source_path"),
                }
                yield _format_sse("error", IngestStatusResponse(**payload).model_dump(mode="json"))
                logger.exception("SSE progress stream failed run_id=%s error=%s", ingest_run_id, exc)
                break
            # 分支2：run 被删了，没有数据
            if state is None:
                break

            status_response = _to_ingest_status_response(state)
            updated_at = state.get("updated_at")

            #分支3：updated_at 变化，推送json
            if last_updated_at is None or updated_at != last_updated_at:
                yield _format_sse("progress", status_response.model_dump(mode="json")) #格式化拼接成 SSE（服务器推送事件）标准报文。
                last_updated_at = updated_at
                last_heartbeat = time.monotonic()
                if status_response.status not in ACTIVE_RUN_STATUSES: #当run状态进入completed / failed / cancelled ，结束sse推送
                    break
            #分支4：15 秒无变化，推送心跳保活防断连
            elif time.monotonic() - last_heartbeat >= 15:
                yield ": heartbeat\n\n"
                last_heartbeat = time.monotonic()

            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# 查询，无生成
@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    resolved_fetch_k = max(request.fetch_k or settings.fetch_k, request.top_k, settings.rerank_candidate_limit) #fetch_k 取三者最大值
    resolved_min_score = settings.min_score if request.min_score is None else request.min_score
    _ensure_indexed_store()
    try:
        results: list[RetrievalResult] = _get_retriever().retrieve(
            query=request.query,
            limit=resolved_fetch_k,
            search_method=request.search_method,
            min_score=resolved_min_score,
            top_k=request.top_k,
        )
    except MilvusOperationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(
        query=request.query,
        search_method=request.search_method,
        top_k=request.top_k,
        fetch_k=resolved_fetch_k,
        min_score=resolved_min_score,
        hits=[
            QueryHit(doc_id=item.doc_id, content=item.content, score=item.score, metadata=item.metadata)
            for item in results #遍历返回召回结果
        ],
    )

#查询，有生成
@app.post("/answer", response_model=AnswerResponse)
def answer_query(request: AnswerRequest) -> AnswerResponse:
    query_response = query_documents(QueryRequest(query=request.query, top_k=request.top_k))  # 执行召回
    retrieval_results = [
        RetrievalResult(doc_id=item.doc_id, content=item.content, score=item.score, metadata=item.metadata)
        for item in query_response.hits
    ]
    #无结果的情况
    if not retrieval_results:
        return AnswerResponse(
            query=request.query,
            answer="抱歉，未检索到相关信息，建议您咨询专业医生或前往医院就诊。",
            fallback=True,
            citations=[],
            retrieval_results=[],
        )
    #有召回结果的情况，generator生成
    generation_result = _get_generator().generate(request.query, retrieval_results)
    #把 LLM 生成答案里的引用编号重新编排，返回整理好后的答案+citations
    normalized_answer, cited_results = normalize_answer_citations(generation_result.answer, retrieval_results)
    # 返回召回结果+citations
    return AnswerResponse(
        query=request.query,
        answer=normalized_answer,
        fallback=generation_result.fallback,
        citations=[
            QueryHit(doc_id=item.doc_id, content=item.content, score=item.score, metadata=item.metadata)
            for item in cited_results
        ],
        retrieval_results=[
            QueryHit(doc_id=item.doc_id, content=item.content, score=item.score, metadata=item.metadata)
            for item in retrieval_results
        ],
    )


from app.web import mount_web

# 挂载其他前端页面路由到app main文件
mount_web(app)
