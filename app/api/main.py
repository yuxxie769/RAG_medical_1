from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.core.logging import get_logger
from app.data import clean_records, iter_raw_records
from app.generation import Generator
from app.kb import build_documents
from app.core.models import RetrievalResult
from app.retrieval import (
    DummyEmbedder,
    HybridRetriever,
    MilvusCollectionSchema,
    MilvusStore,
    OpenAICompatibleEmbedder,
    VectorIndexer,
)

app = FastAPI(title=settings.project_name)
logger = get_logger(__name__)

_DATASTORE = {
    "embedder": None,
    "store": None,
    "indexer": None,
    "generator": None,
    "retriever": None,
}

INGEST_LOG_DIR = Path(os.getenv("INGEST_LOG_DIR", "logs/ingest"))
INGEST_STATE_DIR = Path(os.getenv("INGEST_STATE_DIR", "logs/ingest_state"))


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to raw jsonl file")
    raw_batch_id: str | None = Field(default=None, description="Optional raw batch id")
    clean_batch_id: str | None = Field(default=None, description="Optional clean batch id")
    kb_batch_id: str | None = Field(default=None, description="Optional knowledge base batch id")
    flush_to_milvus: bool = Field(default=True, description="Whether to write into Milvus after indexing")
    resume_from_line: int = Field(default=1, ge=1, description="Resume ingest from this source line number")
    force_reingest: bool = Field(default=False, description="Ignore saved ingest state and start from resume_from_line")


class IngestResponse(BaseModel):
    status: str
    source_path: str
    raw_count: int
    cleaned_count: int
    document_count: int
    milvus_written: bool = False
    milvus_mode: str = "in_memory"
    ingest_run_id: str
    skipped_batches: int = 0
    failed_batches: int = 0
    resumed_from_line: int = 1


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


class QueryResponse(BaseModel):
    query: str
    search_method: Literal["hybrid", "dense", "sparse"]
    top_k: int
    fetch_k: int
    min_score: float
    hits: list[QueryHit] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.top_k, ge=1, le=100)


class AnswerResponse(BaseModel):
    query: str
    answer: str
    fallback: bool = False
    citations: list[QueryHit] = Field(default_factory=list)


class IngestStatusResponse(BaseModel):
    ingest_run_id: str
    exists: bool
    status: str | None = None
    source_path: str | None = None
    raw_count: int = 0
    cleaned_count: int = 0
    document_count: int = 0
    skipped_batches: int = 0
    failed_batches: int = 0
    resume_from_line: int = 1
    last_source_line_no: int | None = None


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "project": settings.project_name}


def _append_ingest_log(payload: dict) -> None:
    INGEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = INGEST_LOG_DIR / f"{payload['ingest_run_id']}.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _save_ingest_state(payload: dict) -> None:
    INGEST_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = INGEST_STATE_DIR / f"{payload['ingest_run_id']}.json"
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# load指点名称json ingest log文件
def _load_ingest_state(ingest_run_id: str) -> dict | None:
    state_path = INGEST_STATE_DIR / f"{ingest_run_id}.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))

# 根据传入的数据源路径，读取最新的ingest状态
def _latest_ingest_state(source_path: str) -> dict | None:
    if not INGEST_STATE_DIR.exists():
        return None
    candidates = []
    for path in INGEST_STATE_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("source_path") == source_path:
            candidates.append(payload)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("last_source_line_no") or 0, reverse=True)
    return candidates[0]


def _get_embedder():
    if settings.embedding_api_key:
        logger.info("Using OpenAI-compatible embedder with remote API.")
        return OpenAICompatibleEmbedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model_name,
            timeout=settings.embedding_timeout_seconds,
        )
    logger.info("Using dummy embedder for local development.")
    return DummyEmbedder(settings.embedding_dimension)


def _get_store() -> MilvusStore:
    if _DATASTORE["store"] is None:
        schema = MilvusCollectionSchema(
            collection_name=settings.milvus_collection_name,
            dimension=settings.embedding_dimension,
            dense_metric_type=settings.milvus_dense_metric_type,
            sparse_vector_field_name=settings.milvus_sparse_field_name,
            analyzer_type=settings.milvus_analyzer_type,
            rrf_k=settings.hybrid_rrf_k,
        )
        logger.info(
            "Initializing Milvus store collection=%s host=%s port=%s",
            settings.milvus_collection_name,
            settings.milvus_host,
            settings.milvus_port,
        )
        _DATASTORE["store"] = MilvusStore(schema=schema, host=settings.milvus_host, port=settings.milvus_port)
    return _DATASTORE["store"]


def _get_indexer():
    if _DATASTORE["indexer"] is None:
        _DATASTORE["embedder"] = _get_embedder()
        _DATASTORE["indexer"] = VectorIndexer(
            embedder=_DATASTORE["embedder"],
            store=_get_store(),
            batch_size=settings.batch_size,
        )
        logger.info("Vector indexer initialized, batch_size=%s", settings.batch_size)
    return _DATASTORE["indexer"]


def _get_retriever() -> HybridRetriever:
    if _DATASTORE["retriever"] is None:
        if _DATASTORE["embedder"] is None:
            _DATASTORE["embedder"] = _get_embedder()
        _DATASTORE["retriever"] = HybridRetriever(
            embedder=_DATASTORE["embedder"],
            store=_get_store(),
        )
    return _DATASTORE["retriever"]


def _get_generator():
    if _DATASTORE["generator"] is None:
        logger.info("Initializing generator model=%s", settings.llm_model_name)
        _DATASTORE["generator"] = Generator(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model_name,
            timeout=settings.llm_timeout_seconds,
        )
    return _DATASTORE["generator"]

# 确保milvu有数据
def _ensure_indexed_store() -> MilvusStore:
    store = _get_store()
    try:
        count = store.count()
    except Exception as exc:
        logger.exception("Failed to query Milvus collection state")
        raise HTTPException(status_code=500, detail=f"Milvus store unavailable: {exc}") from exc

    if count <= 0:
        logger.warning("Query rejected because Milvus collection is empty")
        raise HTTPException(status_code=400, detail="No documents indexed yet. Call /ingest first.")
    logger.info("Milvus collection ready, entity_count=%s", count)
    return store


@app.post("/ingest", response_model=IngestResponse)
def ingest_data(request: IngestRequest) -> IngestResponse:
    ingest_run_id = uuid4().hex # 随机数生成
    logger.info(
        "Ingest started run_id=%s source_path=%s raw_batch_id=%s clean_batch_id=%s kb_batch_id=%s flush_to_milvus=%s resume_from_line=%s",
        ingest_run_id,
        request.source_path,
        request.raw_batch_id,
        request.clean_batch_id,
        request.kb_batch_id,
        request.flush_to_milvus,
        request.resume_from_line,
    )

    raw_count = 0
    cleaned_count = 0
    document_count = 0
    skipped_batches = 0
    failed_batches = 0
    last_source_line_no = 0
    indexer = _get_indexer()
    buffer: list[dict] = [] # 缓存区，用于存储当前批次的记录

    resume_from_line = request.resume_from_line
    # 如果请求本身参数没指定resume_from_line，则加载当前目标ingest文件最新state，查看上次采集到哪行了
    if request.resume_from_line <= 1 and not request.force_reingest:
        latest_state = _latest_ingest_state(request.source_path) #加载当前目标ingest文件最新state
        if latest_state: # 如果存在历史state
            resume_from_line = int(latest_state.get("last_source_line_no") or 0) + 1 # 本次上次采集的行数
            logger.info(
                "Auto resume enabled source_path=%s resume_from_line=%s latest_run_id=%s",
                request.source_path,
                resume_from_line, 
                latest_state.get("ingest_run_id"), 
            )

    try: 
        # 开始采集
        for item in iter_raw_records(request.source_path, batch_id=request.raw_batch_id): # 流式读取数据
            source_line_no = int(item.get("source_line_no", 0)) # 当前行号
            if source_line_no < resume_from_line: # 跳过所有已经采集过的行
                continue

            raw_count += 1
            buffer.append(item) # 缓存区添加当前行
            last_source_line_no = source_line_no

            if len(buffer) < settings.batch_size:
                continue

            try:
                # 当一批数量攒够，清洗当前批次，生成文档，存入milvus
                cleaned_records = clean_records(buffer, clean_batch_id=request.clean_batch_id)
                cleaned_count += len(cleaned_records)
                documents = build_documents(cleaned_records, batch_id=request.kb_batch_id)
                document_count += len(documents)

                if request.flush_to_milvus and documents:
                    result = indexer.index_in_batches(documents) # 存入milvus
                    # 统计跳过和失败的数量
                    if result.skipped_count > 0:
                        skipped_batches += 1
                    failed_batches += result.failed_batches
                    logger.info(
                        "Batch indexed run_id=%s source_line_no=%s raw_count=%s cleaned_count=%s document_count=%s indexed=%s skipped=%s failed=%s",
                        ingest_run_id,
                        last_source_line_no,
                        raw_count,
                        cleaned_count,
                        document_count,
                        result.indexed_count,
                        result.skipped_count,
                        result.failed_batches,
                    )
                else:
                    logger.info("Batch processed but not flushed run_id=%s document_count=%s", ingest_run_id, len(documents))
                
                # 每更新一个batch就更新ingest log、状态
                _append_ingest_log(
                    {
                        "ingest_run_id": ingest_run_id,
                        "source_path": request.source_path,
                        "raw_count": raw_count,
                        "cleaned_count": cleaned_count,
                        "document_count": document_count,
                        "skipped_batches": skipped_batches,
                        "failed_batches": failed_batches,
                        "resume_from_line": resume_from_line,
                        "last_source_line_no": last_source_line_no,
                        "status": "running",
                    }
                )
                _save_ingest_state(
                    {
                        "ingest_run_id": ingest_run_id,
                        "source_path": request.source_path,
                        "raw_batch_id": request.raw_batch_id,
                        "clean_batch_id": request.clean_batch_id,
                        "kb_batch_id": request.kb_batch_id,
                        "raw_count": raw_count,
                        "cleaned_count": cleaned_count,
                        "document_count": document_count,
                        "resume_from_line": resume_from_line,
                        "last_source_line_no": last_source_line_no,
                        "skipped_batches": skipped_batches,
                        "failed_batches": failed_batches,
                        "flush_to_milvus": request.flush_to_milvus,
                        "force_reingest": request.force_reingest,
                    }
                )
            except Exception as exc:
                failed_batches += 1
                logger.exception("Batch failed and skipped run_id=%s error=%s", ingest_run_id, exc)
                _append_ingest_log(
                    {
                        "ingest_run_id": ingest_run_id,
                        "source_path": request.source_path,
                        "raw_count": raw_count,
                        "cleaned_count": cleaned_count,
                        "document_count": document_count,
                        "skipped_batches": skipped_batches,
                        "failed_batches": failed_batches,
                        "resume_from_line": resume_from_line,
                        "last_source_line_no": last_source_line_no,
                        "status": "batch_failed",
                        "error": str(exc),
                    }
                )
            finally:
                buffer = [] # 清空缓存区
        
        # 结束流式读取，处理buffer里面最后一批数据
        if buffer: 
            try:
                cleaned_records = clean_records(buffer, clean_batch_id=request.clean_batch_id)
                cleaned_count += len(cleaned_records)
                documents = build_documents(cleaned_records, batch_id=request.kb_batch_id)
                document_count += len(documents)
                if request.flush_to_milvus and documents:
                    result = indexer.index_in_batches(documents)
                    if result.skipped_count > 0:
                        skipped_batches += 1
                    failed_batches += result.failed_batches
                    logger.info(
                        "Tail batch indexed run_id=%s source_line_no=%s indexed=%s skipped=%s failed=%s",
                        ingest_run_id,
                        last_source_line_no,
                        result.indexed_count,
                        result.skipped_count,
                        result.failed_batches,
                    )
            except Exception as exc:
                failed_batches += 1
                logger.exception("Tail batch failed and skipped run_id=%s error=%s", ingest_run_id, exc)

        # 结束采集，更新ingest log、状态
        logger.info("Ingest finished run_id=%s source_path=%s", ingest_run_id, request.source_path)
        _append_ingest_log(
            {
                "ingest_run_id": ingest_run_id,
                "source_path": request.source_path,
                "raw_count": raw_count,
                "cleaned_count": cleaned_count,
                "document_count": document_count,
                "skipped_batches": skipped_batches,
                "failed_batches": failed_batches,
                "resume_from_line": resume_from_line,
                "last_source_line_no": last_source_line_no,
                "status": "finished",
            }
        )
        _save_ingest_state(
            {
                "ingest_run_id": ingest_run_id,
                "source_path": request.source_path,
                "raw_batch_id": request.raw_batch_id,
                "clean_batch_id": request.clean_batch_id,
                "kb_batch_id": request.kb_batch_id,
                "raw_count": raw_count,
                "cleaned_count": cleaned_count,
                "document_count": document_count,
                "resume_from_line": resume_from_line,
                "last_source_line_no": last_source_line_no,
                "skipped_batches": skipped_batches,
                "failed_batches": failed_batches,
                "flush_to_milvus": request.flush_to_milvus,
                "force_reingest": request.force_reingest,
                "status": "finished",
            }
        )
        return IngestResponse(
            status="ok",
            source_path=request.source_path,
            raw_count=raw_count,
            cleaned_count=cleaned_count,
            document_count=document_count,
            milvus_written=request.flush_to_milvus and document_count > 0,
            milvus_mode="milvus",
            ingest_run_id=ingest_run_id,
            skipped_batches=skipped_batches,
            failed_batches=failed_batches,
            resumed_from_line=resume_from_line,
        )
    except Exception as exc:
        logger.exception("Ingest failed run_id=%s error=%s", ingest_run_id, exc)
        _append_ingest_log(
            {
                "ingest_run_id": ingest_run_id,
                "source_path": request.source_path,
                "raw_count": raw_count,
                "cleaned_count": cleaned_count,
                "document_count": document_count,
                "skipped_batches": skipped_batches,
                "failed_batches": failed_batches,
                "resume_from_line": resume_from_line,
                "last_source_line_no": last_source_line_no,
                "status": "failed",
                "error": str(exc),
            }
        )
        _save_ingest_state(
            {
                "ingest_run_id": ingest_run_id,
                "source_path": request.source_path,
                "raw_batch_id": request.raw_batch_id,
                "clean_batch_id": request.clean_batch_id,
                "kb_batch_id": request.kb_batch_id,
                "raw_count": raw_count,
                "cleaned_count": cleaned_count,
                "document_count": document_count,
                "resume_from_line": resume_from_line,
                "last_source_line_no": last_source_line_no,
                "skipped_batches": skipped_batches,
                "failed_batches": failed_batches,
                "flush_to_milvus": request.flush_to_milvus,
                "force_reingest": request.force_reingest,
                "status": "failed",
                "error": str(exc),
            }
        )
        raise

# 访问指定ingest run_id的采集状态
@app.get("/ingest/status", response_model=IngestStatusResponse)
def ingest_status(ingest_run_id: str) -> IngestStatusResponse:
    state = _load_ingest_state(ingest_run_id)
    if state is None:
        return IngestStatusResponse(ingest_run_id=ingest_run_id, exists=False)
    return IngestStatusResponse(
        ingest_run_id=ingest_run_id,
        exists=True,
        status=state.get("status"),
        source_path=state.get("source_path"),
        raw_count=int(state.get("raw_count", 0)),
        cleaned_count=int(state.get("cleaned_count", 0)),
        document_count=int(state.get("document_count", 0)),
        skipped_batches=int(state.get("skipped_batches", 0)),
        failed_batches=int(state.get("failed_batches", 0)),
        resume_from_line=int(state.get("resume_from_line", 1)),
        last_source_line_no=state.get("last_source_line_no"),
    )

# 
@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    resolved_fetch_k = max(request.fetch_k or settings.fetch_k, request.top_k)
    resolved_min_score = settings.min_score if request.min_score is None else request.min_score
    logger.info(
        "Query started query=%s search_method=%s top_k=%s fetch_k=%s min_score=%s",
        request.query,
        request.search_method,
        request.top_k,
        resolved_fetch_k,
        resolved_min_score,
    )
    store = _ensure_indexed_store()

    retriever = _get_retriever()
    results: list[RetrievalResult] = retriever.retrieve(
        query=request.query,
        limit=resolved_fetch_k,
        search_method=request.search_method,
    )
    filtered_results = [item for item in results if item.score >= resolved_min_score][: request.top_k]
    logger.info("Query retrieved hits=%s filtered_hits=%s", len(results), len(filtered_results))

    hits = [
        QueryHit(doc_id=item.doc_id, content=item.content, score=item.score, metadata=item.metadata)
        for item in filtered_results
    ]
    return QueryResponse(
        query=request.query,
        search_method=request.search_method,
        top_k=request.top_k,
        fetch_k=resolved_fetch_k,
        min_score=resolved_min_score,
        hits=hits,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer_query(request: AnswerRequest) -> AnswerResponse:
    logger.info("Answer request started query=%s top_k=%s", request.query, request.top_k)
    query_response = query_documents(QueryRequest(query=request.query, top_k=request.top_k))
    citations = [
        RetrievalResult(
            doc_id=item.doc_id,
            content=item.content,
            score=item.score,
            metadata=item.metadata,
        )
        for item in query_response.hits
    ]
    generator = _get_generator()
    generation_result = generator.generate(request.query, citations)

    citation_hits = [
        QueryHit(doc_id=item.doc_id, content=item.content, score=item.score, metadata=item.metadata)
        for item in citations
    ]
    logger.info("Answer generation finished fallback=%s", generation_result.fallback)
    return AnswerResponse(
        query=request.query,
        answer=generation_result.answer,
        fallback=generation_result.fallback,
        citations=citation_hits,
    )
