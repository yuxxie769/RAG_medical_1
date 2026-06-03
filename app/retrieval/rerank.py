from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.core.models import RetrievalResult


@dataclass
class RerankResult:
    doc_id: str
    score: float


class Reranker(Protocol):
    provider_name: str

    def rerank(self, query: str, results: Sequence[RetrievalResult]) -> list[RerankResult]:
        raise NotImplementedError


@dataclass
class NoopReranker:
    provider_name: str = "none"

    def rerank(self, query: str, results: Sequence[RetrievalResult]) -> list[RerankResult]:
        return [RerankResult(doc_id=result.doc_id, score=float(result.score)) for result in results]


@dataclass
class OpenAICompatibleReranker:
    api_key: str
    base_url: str
    model: str
    path: str = "/rerank"
    timeout: float = 10.0
    provider_name: str = "openai_compatible"

    def rerank(self, query: str, results: Sequence[RetrievalResult]) -> list[RerankResult]:
        if not self.base_url:
            raise RuntimeError("RERANK_BASE_URL is empty")
        if not self.api_key:
            raise RuntimeError("RERANK_API_KEY is empty")
        if not self.model:
            raise RuntimeError("RERANK_MODEL_NAME is empty")

        endpoint = f"{self.base_url.rstrip('/')}/{self.path.lstrip('/')}"
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "documents": [result.content for result in results],
                "top_n": len(results),
            }
        ).encode("utf-8")
        http_request = urllib_request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Rerank HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Rerank request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Rerank request timed out") from exc

        raw_results = response_payload.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("Rerank response missing results list")

        reranked: list[RerankResult] = []
        seen_indices: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise RuntimeError("Rerank response contains non-object result")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int):
                raise RuntimeError("Rerank response result index is invalid")
            if index < 0 or index >= len(results):
                raise RuntimeError(f"Rerank response index out of range: {index}")
            if index in seen_indices:
                raise RuntimeError(f"Rerank response contains duplicate index: {index}")
            try:
                score_value = float(score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Rerank response relevance_score is invalid") from exc
            seen_indices.add(index)
            reranked.append(RerankResult(doc_id=results[index].doc_id, score=score_value))

        if len(reranked) != len(results):
            raise RuntimeError("Rerank response does not cover every candidate")
        return reranked
