from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from app.core.models import Document, RetrievalResult


@dataclass
class RetrievalConfig:
    top_k: int = 5
    min_score: float = 0.0


@dataclass
class Retriever:
    """Base retriever interface.

    The retriever should be responsible only for fetching relevant documents.
    It should not contain generation logic.
    """

    documents: Sequence[Document] = field(default_factory=list)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """Return top-k retrieval results for a query."""
        raise NotImplementedError
