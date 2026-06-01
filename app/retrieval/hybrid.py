from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.core.models import RetrievalResult
from app.retrieval.embedder import Embedder
from app.retrieval.milvus_store import MilvusStore, SearchMethod
from app.retrieval.schemas import HybridRetrievalHit, HybridRetrievalRequest


@dataclass
class HybridRetriever:
    """Dispatch dense, sparse, or Milvus-native hybrid retrieval."""

    embedder: Embedder
    store: MilvusStore

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_method: SearchMethod = "hybrid",
    ) -> List[RetrievalResult]:
        query_vector = None
        if search_method in {"hybrid", "dense"}:
            query_vector = self.embedder.embed_texts([query])[0]
        return self.store.search(
            query_vector=query_vector,
            query_text=query,
            limit=limit,
            search_method=search_method,
        )


@dataclass
class RRFConfig:
    k: int = 60


@dataclass
class WeightedFusionConfig:
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    rerank_weight: float = 0.2


def fuse_scores(
    dense_scores: Dict[str, float],
    sparse_scores: Dict[str, float],
    rerank_scores: Optional[Dict[str, float]] = None,
    config: Optional[WeightedFusionConfig] = None,
) -> Dict[str, float]:
    """Simple weighted fusion for dense + sparse + rerank scores."""
    cfg = config or WeightedFusionConfig()
    rerank_scores = rerank_scores or {}
    fused: Dict[str, float] = {}
    doc_ids = set(dense_scores) | set(sparse_scores) | set(rerank_scores)
    for doc_id in doc_ids:
        fused[doc_id] = (
            dense_scores.get(doc_id, 0.0) * cfg.dense_weight
            + sparse_scores.get(doc_id, 0.0) * cfg.sparse_weight
            + rerank_scores.get(doc_id, 0.0) * cfg.rerank_weight
        )
    return fused


def reorder_hits(hits: Sequence[HybridRetrievalHit]) -> List[HybridRetrievalHit]:
    """Order hits by score descending."""
    return sorted(hits, key=lambda hit: hit.score, reverse=True)


def to_hybrid_hits(results: Sequence[RetrievalResult]) -> List[HybridRetrievalHit]:
    return [
        HybridRetrievalHit(
            doc_id=result.doc_id,
            content=result.content,
            score=result.score,
            dense_score=result.score,
            sparse_score=0.0,
            rerank_score=0.0,
            metadata=result.metadata,
        )
        for result in results
    ]
