from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from openai import OpenAI

from app.retrieval.embedder import Embedder


@dataclass
class QwenOpenAIEmbedder(Embedder):
    """OpenAI-compatible Qwen Cloud embedding adapter.

    Docs: https://docs.qwencloud.com/api-reference/text-embedding/openai-embedding
    """

    api_key: str
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model: str = "text-embedding-v4"
    dimensions: int = 1024
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=self.dimensions,
            encoding_format="float",
        )

        embeddings: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        for item in response.data:
            embeddings[item.index] = list(item.embedding)
        return embeddings
