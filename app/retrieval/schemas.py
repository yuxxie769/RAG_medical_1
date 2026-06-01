from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorRecord:
    doc_id: str
    vector: List[float]
    content: str
    question: str
    answer: str
    metadata: Dict[str, Any]


@dataclass
class HybridRetrievalRequest:
    query: str
    top_k: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    rerank_weight: float = 0.2
    source_filter: Optional[str] = None
    split_filter: Optional[str] = None


@dataclass
class HybridRetrievalHit:
    doc_id: str
    content: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
