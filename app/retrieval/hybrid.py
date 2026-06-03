from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence

from app.core.logging import get_logger
from app.core.models import RetrievalResult
from app.retrieval.embedder import Embedder
from app.retrieval.milvus_store import MilvusStore, SearchMethod
from app.retrieval.rerank import NoopReranker, Reranker
from app.retrieval.schemas import HybridRetrievalHit, HybridRetrievalRequest


logger = get_logger(__name__)


@dataclass
class HybridRetriever:
    """Dispatch dense, sparse, or Milvus-native hybrid retrieval."""

    embedder: Embedder
    store: MilvusStore
    reranker: Reranker = field(default_factory=NoopReranker)
    rerank_candidate_limit: int = 15

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_method: SearchMethod = "hybrid",
        min_score: float = 0.0,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        query_vector = None
        if search_method in {"hybrid", "dense"}:
            query_vector = self.embedder.embed_texts([query])[0]
        results = self.store.search(
            query_vector=query_vector,
            query_text=query,
            limit=limit,
            search_method=search_method,
        )
        filtered_results = [result for result in results if float(result.score) >= float(min_score)]
        resolved_top_k = top_k or limit
        rerank_limit = max(0, int(self.rerank_candidate_limit))
        provider_name = self.reranker.provider_name

        if provider_name == "none" or len(filtered_results) < 2 or rerank_limit < 2:
            return self._annotate_without_rerank(filtered_results[:resolved_top_k], provider_name)

        rerank_candidates = filtered_results[:rerank_limit]
        remaining_candidates = filtered_results[rerank_limit:]
        rerank_started = perf_counter()
        try:
            reranked = self.reranker.rerank(query, rerank_candidates)
            rerank_seconds = perf_counter() - rerank_started
            rerank_scores = {item.doc_id: item.score for item in reranked}
            if len(rerank_scores) != len(rerank_candidates):
                raise RuntimeError("Rerank response did not produce a score for every candidate")
            ranked_candidates = sorted(
                rerank_candidates,
                key=lambda result: (rerank_scores[result.doc_id], float(result.score)),
                reverse=True,
            )
            logger.info(
                "rerank provider=%s candidate_count=%s rerank_seconds=%.3f fallback=%s",
                provider_name,
                len(rerank_candidates),
                rerank_seconds,
                False,
            )
            final_results = [
                self._copy_result(
                    result,
                    score=rerank_scores[result.doc_id],
                    rerank_provider=provider_name,
                    rerank_score=rerank_scores[result.doc_id],
                    rerank_applied=True,
                )
                for result in ranked_candidates
            ]
            final_results.extend(self._annotate_without_rerank(remaining_candidates, provider_name))
            return final_results[:resolved_top_k]
        except Exception as exc:
            rerank_seconds = perf_counter() - rerank_started
            logger.warning(
                "rerank provider=%s candidate_count=%s rerank_seconds=%.3f fallback=%s error=%s",
                provider_name,
                len(rerank_candidates),
                rerank_seconds,
                True,
                exc,
            )
            return self._annotate_without_rerank(filtered_results[:resolved_top_k], provider_name)

    def _annotate_without_rerank(
        self,
        results: Sequence[RetrievalResult],
        provider_name: str,
    ) -> List[RetrievalResult]:
        return [
            self._copy_result(
                result,
                score=float(result.score),
                rerank_provider=provider_name,
                rerank_score=None,
                rerank_applied=False,
            )
            for result in results
        ]

    @staticmethod
    def _copy_result(
        result: RetrievalResult,
        *,
        score: float,
        rerank_provider: str,
        rerank_score: float | None,
        rerank_applied: bool,
    ) -> RetrievalResult:
        metadata = dict(result.metadata)
        metadata["retrieval_score"] = float(result.score)
        metadata["rerank_score"] = rerank_score
        metadata["rerank_applied"] = rerank_applied
        metadata["rerank_provider"] = rerank_provider
        return RetrievalResult(
            doc_id=result.doc_id,
            content=result.content,
            score=float(score),
            metadata=metadata,
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
            dense_score=float(result.metadata.get("retrieval_score", result.score)),
            sparse_score=0.0,
            rerank_score=float(result.metadata.get("rerank_score", 0.0) or 0.0),
            metadata=result.metadata,
        )
        for result in results
    ]
