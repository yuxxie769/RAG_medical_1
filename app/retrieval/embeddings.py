from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    """Embedding interface for text-to-vector conversion."""

    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...


@dataclass
class EmbedderConfig:
    model_name: str = "unknown"
    dimension: int = 0
    batch_size: int = 32
