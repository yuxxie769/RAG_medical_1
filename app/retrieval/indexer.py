from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Literal, Sequence

from app.core.logging import get_logger
from app.core.models import Document
from app.retrieval.embedder import Embedder
from app.retrieval.milvus_store import MilvusStore, build_records


logger = get_logger(__name__)


@dataclass
class VectorIndexingResult:
    batch_size: int
    indexed_count: int
    existing_count: int = 0
    failed_count: int = 0
    failed_batches: int = 0
    milvus_failed_batches: int = 0
    embedding_seconds: float = 0.0
    failure_stage: Literal["embedding", "milvus"] | None = None
    errors: List[str] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return self.failed_count


@dataclass
class VectorIndexer:
    embedder: Embedder
    store: MilvusStore
    batch_size: int = 128

    def index_documents(self, documents: Sequence[Document]) -> VectorIndexingResult:
        try:
            embedding_started = perf_counter()
            vectors = self.embedder.embed_texts([doc.content for doc in documents])
            embedding_seconds = perf_counter() - embedding_started
            records = build_records(documents, vectors)
        except Exception as exc:
            logger.exception("Failed to embed document batch: %s", exc)
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=0,
                failed_count=len(documents),
                failed_batches=1,
                failure_stage="embedding",
                errors=[str(exc)],
            )

        try:
            self.store.create_collection()
            result = self.store.upsert(records)
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=result.inserted_count,
                existing_count=result.existing_count,
                embedding_seconds=embedding_seconds,
            )
        except Exception as exc:
            logger.exception("Failed to write document batch to Milvus: %s", exc)
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=0,
                failed_count=len(documents),
                failed_batches=1,
                milvus_failed_batches=1,
                embedding_seconds=embedding_seconds,
                failure_stage="milvus",
                errors=[str(exc)],
            )

    def index_in_batches(self, documents: Sequence[Document]) -> VectorIndexingResult:
        total = VectorIndexingResult(batch_size=self.batch_size, indexed_count=0)
        batch: List[Document] = []
        for document in documents:
            batch.append(document)
            if len(batch) >= self.batch_size:
                self._merge(total, self.index_documents(batch))
                batch = []
        if batch:
            self._merge(total, self.index_documents(batch))
        return total

    def flush(self) -> None:
        self.store.flush()

    def verify_persisted_doc_ids(self, doc_ids: Sequence[str]) -> None:
        self.store.verify_persisted_doc_ids(doc_ids)

    @staticmethod
    def _merge(total: VectorIndexingResult, result: VectorIndexingResult) -> None:
        total.indexed_count += result.indexed_count
        total.existing_count += result.existing_count
        total.failed_count += result.failed_count
        total.failed_batches += result.failed_batches
        total.milvus_failed_batches += result.milvus_failed_batches
        total.embedding_seconds += result.embedding_seconds
        if total.failure_stage is None and result.failure_stage is not None:
            total.failure_stage = result.failure_stage
        total.errors.extend(result.errors)
