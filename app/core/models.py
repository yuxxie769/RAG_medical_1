from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Knowledge base document used for indexing and retrieval."""

    doc_id: str
    question: str
    answer: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A single retrieval hit returned by the retriever."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Final answer produced by the generation module."""

    query: str
    answer: str
    citations: List[RetrievalResult] = Field(default_factory=list)
    fallback: bool = False
    latency: Optional[float] = None


class EvaluationResult(BaseModel):
    """Evaluation metrics for retrieval/generation."""

    recall_at_5: float
    recall_at_10: float
    mrr: float
    sample_count: int
