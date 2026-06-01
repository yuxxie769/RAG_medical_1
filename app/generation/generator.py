from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from openai import OpenAI

from app.core.logging import get_logger
from app.core.models import GenerationResult, RetrievalResult

logger = get_logger(__name__)


@dataclass
class Generator:
    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0
    system_prompt: str = (
        "你是一个严谨的医疗问答助手。你必须只根据提供的检索证据回答，"
        "如果证据不足，请明确说明无法确认，并建议用户就医或进一步检查。"
    )
    client: OpenAI | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            logger.warning("LLM api key is empty, generator will use fallback mode")
            return
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _build_prompt(self, query: str, citations: Sequence[RetrievalResult]) -> str:
        evidence_lines: List[str] = []
        for idx, item in enumerate(citations, start=1):
            evidence_lines.append(f"[{idx}] {item.content}")
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "无可用证据"
        return (
            f"问题：{query}\n\n"
            f"检索证据：\n{evidence_text}\n\n"
            "请基于以上证据回答，要求：\n"
            "1. 只使用证据中的信息\n"
            "2. 不确定时明确说明\n"
            "3. 输出简洁、专业、适合医疗场景\n"
        )

    def generate(self, query: str, citations: Sequence[RetrievalResult]) -> GenerationResult:
        if not self.api_key or self.client is None:
            logger.warning("LLM api key is empty, returning fallback answer")
            return GenerationResult(
                query=query,
                answer="当前未配置 LLM API Key，暂无法生成答案。",
                citations=list(citations),
                fallback=True,
            )

        prompt = self._build_prompt(query, citations)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.timeout,
            )
            answer = response.choices[0].message.content or ""
            return GenerationResult(
                query=query,
                answer=answer.strip(),
                citations=list(citations),
                fallback=False,
            )
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            return GenerationResult(
                query=query,
                answer="生成失败，请稍后重试或检查模型配置。",
                citations=list(citations),
                fallback=True,
            )
