from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence

from openai import OpenAI


class Embedder(Protocol):
    def embed_text(self, text: str) -> List[float]:
        """Return a vector for a single text."""
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Return vectors for a batch of texts."""
        raise NotImplementedError


@dataclass
class OpenAICompatibleEmbedder:
    api_key: str
    base_url: str
    model: str
    dimensions: int = 1024
    timeout: float = 60.0
    # 用open ai 通用接口加载emb模型，接受一个批次的文档内容，串行处理后返回向量list
    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return list(response.data[0].embedding)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


@dataclass
class DummyEmbedder:
    """Development stub for embedding interface."""

    dimension: int = 8

    def embed_text(self, text: str) -> List[float]:
        seed = float(len(text) % 10)
        return [seed for _ in range(self.dimension)]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]
