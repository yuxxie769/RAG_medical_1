from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from app.core.logging import get_logger
from app.core.models import Document
from app.retrieval.embedder import Embedder
from app.retrieval.milvus_store import MilvusRecord, MilvusStore, build_records


logger = get_logger(__name__)


@dataclass
class VectorIndexingResult:
    batch_size: int
    indexed_count: int
    skipped_count: int = 0
    failed_batches: int = 0

# 文档转为向量并存入 Milvus 向量数据库，支持批量分批处理
@dataclass
class VectorIndexer:
    embedder: Embedder
    store: MilvusStore
    batch_size: int = 128

    # 单批存入milvus逻辑
    def index_documents(self, documents: Sequence[Document]) -> VectorIndexingResult:
        try:
            texts = [doc.content for doc in documents] # 获取文档内容list
            vectors = self.embedder.embed_texts(texts) # 转为向量list
            records = build_records(documents, vectors)
            self.store.create_collection()
            self.store.upsert(records) # 一批存入milvus
            return VectorIndexingResult(batch_size=len(documents), indexed_count=len(records))
        except Exception as exc:
            logger.exception("Failed to index document batch, skip this batch: %s", exc)
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=0,
                skipped_count=len(documents),
                failed_batches=1,
            )

    # 分批入口，分批存入milvus
    def index_in_batches(self, documents: Sequence[Document]) -> VectorIndexingResult:
        total_indexed = 0 # 已存入的文档数量
        total_skipped = 0
        failed_batches = 0 # 失败的分批数量
        batch: List[Document] = []

        for document in documents:
            batch.append(document)
            if len(batch) >= self.batch_size: # 达到batch size就开始存
                result = self.index_documents(batch)
                total_indexed += result.indexed_count # 已存入的文档数量
                total_skipped += result.skipped_count # 跳过的文档数量
                failed_batches += result.failed_batches # 失败的批数量
                batch = []
    
        # 最后一批不足batch size的文档，也存入milvus
        if batch:
            result = self.index_documents(batch)
            total_indexed += result.indexed_count
            total_skipped += result.skipped_count
            failed_batches += result.failed_batches

        return VectorIndexingResult(
            batch_size=self.batch_size,
            indexed_count=total_indexed,
            skipped_count=total_skipped,
            failed_batches=failed_batches,
        )
