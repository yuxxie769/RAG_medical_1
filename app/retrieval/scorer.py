from __future__ import annotations

from typing import Protocol

from app.core.models import Document


class ScoringStrategy(Protocol):
    def score(self, query: str, document: Document) -> float:
        ...
