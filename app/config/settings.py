from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"} #匹配成功返回 True，否则返回 False


class Settings(BaseModel):
    project_name: str = os.getenv("PROJECT_NAME", "RAG Medical")
    environment: str = os.getenv("ENVIRONMENT", "dev")
    top_k: int = int(os.getenv("TOP_K", "5"))
    fetch_k: int = int(os.getenv("FETCH_K", "15"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.0"))
    search_method: Literal["hybrid", "dense", "sparse"] = os.getenv("SEARCH_METHOD", "hybrid")
    retrieval_timeout_seconds: int = int(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "3"))
    generation_timeout_seconds: int = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "15"))
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "rag_medical_documents")
    milvus_dense_metric_type: str = os.getenv("MILVUS_DENSE_METRIC_TYPE", "COSINE").upper()
    milvus_sparse_field_name: str = os.getenv("MILVUS_SPARSE_FIELD_NAME", "sparse_embedding")
    milvus_analyzer_type: str = os.getenv("MILVUS_ANALYZER_TYPE", "chinese")
    hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", "60"))
    milvus_request_timeout_seconds: float = float(
        os.getenv("MILVUS_REQUEST_TIMEOUT_SECONDS", str(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "3")))
    )
    milvus_management_timeout_seconds: float = float(
        os.getenv("MILVUS_MANAGEMENT_TIMEOUT_SECONDS", "10")
    )
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv(
        "EMBEDDING_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_timeout_seconds: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
    enable_rerank: bool = _get_bool("ENABLE_RERANK", False)
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "none")
    rerank_api_key: str = os.getenv("RERANK_API_KEY", "")
    rerank_base_url: str = os.getenv("RERANK_BASE_URL", "")
    rerank_path: str = os.getenv("RERANK_PATH", "/rerank")
    rerank_model_name: str = os.getenv("RERANK_MODEL_NAME", "")
    rerank_timeout_seconds: float = float(os.getenv("RERANK_TIMEOUT_SECONDS", "10"))
    rerank_candidate_limit: int = int(os.getenv("RERANK_CANDIDATE_LIMIT", "15"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "128"))
    enable_milvus: bool = _get_bool("ENABLE_MILVUS", False)
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "rag_medical")
    mongodb_connect_timeout_ms: int = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "3000"))
    ingest_lease_seconds: int = int(os.getenv("INGEST_LEASE_SECONDS", "300"))
    admin_bootstrap_username: str = os.getenv("ADMIN_BOOTSTRAP_USERNAME", "")
    admin_bootstrap_password: str = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "")
    admin_session_secret: str = os.getenv("ADMIN_SESSION_SECRET", "dev-admin-session-secret")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    judge_api_key: str = os.getenv("JUDGE_API_KEY", "")
    judge_base_url: str = os.getenv("JUDGE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    judge_model_name: str = os.getenv("JUDGE_MODEL_NAME", "qwen-plus")
    judge_timeout_seconds: int = int(os.getenv("JUDGE_TIMEOUT_SECONDS", "30"))


settings = Settings()
