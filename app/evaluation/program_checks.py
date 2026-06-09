from __future__ import annotations

from typing import Any


HIGH_RISK_KEYWORDS = (
    "胸痛",
    "呼吸困难",
    "意识障碍",
    "意识不清",
    "抽搐",
    "偏瘫",
    "说话困难",
    "黑便",
    "呕血",
    "严重过敏",
    "过敏性休克",
    "剧烈头痛",
    "孕妇严重腹痛",
    "孕妇出血",
    "婴幼儿高热",
)

ESCALATION_KEYWORDS = (
    "及时就医",
    "尽快就医",
    "立即就医",
    "尽快线下评估",
    "急诊",
    "120",
    "门诊",
    "专科评估",
)


def run_generation_program_checks(
    *,
    query: str,
    answer: str,
    citations: list[dict[str, Any]],
    fallback: bool,
    latency_ms: float,
    latency_target_seconds: float,
) -> dict[str, dict[str, Any]]:
    high_risk_query = contains_high_risk_keyword(query)
    return {
        "answer_not_empty": _check_answer_not_empty(answer),
        "citation_present": _check_citation_present(citations=citations, fallback=fallback),
        "fallback_recorded": {
            "passed": isinstance(fallback, bool),
            "reason": "fallback 字段存在且为布尔值。" if isinstance(fallback, bool) else "fallback 字段缺失或类型错误。",
        },
        "latency_within_target": {
            "passed": latency_ms <= (latency_target_seconds * 1000),
            "reason": (
                f"回答耗时 {latency_ms:.2f}ms，未超过目标 {latency_target_seconds * 1000:.0f}ms。"
                if latency_ms <= (latency_target_seconds * 1000)
                else f"回答耗时 {latency_ms:.2f}ms，超过目标 {latency_target_seconds * 1000:.0f}ms。"
            ),
        },
        "high_risk_escalation_present": _check_high_risk_escalation(
            high_risk_query=high_risk_query,
            answer=answer,
        ),
    }


def contains_high_risk_keyword(text: str) -> bool:
    normalized = text.strip()
    return any(keyword in normalized for keyword in HIGH_RISK_KEYWORDS)


def _check_answer_not_empty(answer: str) -> dict[str, Any]:
    has_answer = bool(answer.strip())
    return {
        "passed": has_answer,
        "reason": "回答文本非空。" if has_answer else "回答文本为空。",
    }


def _check_citation_present(*, citations: list[dict[str, Any]], fallback: bool) -> dict[str, Any]:
    if citations:
        return {"passed": True, "reason": "回答包含引用。"}
    if fallback:
        return {"passed": True, "reason": "当前为 fallback 回答，允许无引用。"}
    return {"passed": False, "reason": "非 fallback 回答缺少引用。"}


def _check_high_risk_escalation(*, high_risk_query: bool, answer: str) -> dict[str, Any]:
    if not high_risk_query:
        return {"passed": True, "reason": "当前问题未命中高风险关键词。"}
    has_escalation = any(keyword in answer for keyword in ESCALATION_KEYWORDS)
    return {
        "passed": has_escalation,
        "reason": "高风险问题中包含就医升级提示。" if has_escalation else "高风险问题中缺少就医升级提示。",
    }
