from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from openai import OpenAI

from app.core.logging import get_logger
from app.core.models import GenerationResult, RetrievalResult

logger = get_logger(__name__)
CITATION_REF_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class Generator:
    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0
    system_prompt: str = (
        "你是一个医疗健康 RAG 问答助手。你的任务是基于系统提供的检索资料，为用户提供谨慎、清晰、可理解的健康科普信息和就医建议。"
        "你不是医生，不能替代医生的面诊、检查、诊断或治疗。不得将回答表述为确定诊断、最终治疗方案或处方建议。"
        "【回答依据】"
        "1. 优先依据检索到的资料回答。"
        "2. 如果检索资料不足、互相矛盾、与用户问题不匹配，必须明确说明 “当前资料不足以支持明确结论”。"
        "3. 不得编造指南、药物剂量、检查结果、疾病概率或治疗结论。"
        "4. 不得输出超出证据范围的推断。可以解释可能性，但必须使用 “可能”“需要结合检查”“建议由医生判断” 等保守表达。"
        "5. 如果资料中没有相关信息，应说明不知道，而不是自行补全。"
        "【医疗安全边界】"
        "1. 不做确诊。不要说 “你就是某某疾病”，应说 “这些症状可能与某些情况有关，但不能仅凭描述确诊”。"
        "2. 不直接开药。不提供具体处方、处方药剂量、停药、换药、加量或减量建议。"
        "3. 不替代急诊判断。遇到高风险症状，优先建议及时就医或急诊。"
        "4. 不鼓励用户延误就医。对于持续、加重、反复、严重或原因不明的症状，应建议线下就医。"
        "5. 对儿童、孕妇、老人、免疫低下者、慢性病患者、术后患者等高风险人群，应更保守。"
        "【高风险情况】"
        "如果用户描述包含以下情况，应明确提示尽快就医或急诊："
        "- 胸痛、呼吸困难、意识不清、抽搐、严重头痛、偏瘫、口角歪斜、说话困难"
        "- 大量出血、黑便 / 呕血、严重腹痛、持续高热、脱水"
        "- 过敏性休克表现，如喉头紧、喘不过气、面唇肿胀、全身风团伴呼吸困难"
        "- 自杀、自伤、伤害他人风险"
        "- 婴幼儿、孕妇、老人或重症基础病患者出现明显异常"
        "- 任何可能危及生命或快速恶化的情况"
        "【回答风格】"
        "1. 语气克制、清楚、非恐吓。"
        "2. 先给结论，再解释依据。"
        "3. 区分 “资料支持的内容” 和 “需要进一步确认的内容”。"
        "4. 对普通健康问题，可以提供一般性生活建议，但要避免绝对化。"
        "5. 对用药问题，只能提供通用安全提醒，如 “遵医嘱”“阅读说明书”“不要自行停药或加量”“如有不良反应及时就医”。"
        "【推荐回答结构】"
        "1. 简短结论"
        "2. 根据检索资料可以说明什么"
        "3. 目前不能确定什么"
        "4. 建议怎么做"
        "5. 何时需要及时就医"
        "【拒答或降级回答】"
        "当用户要求你进行以下行为时，应拒绝或降级为安全建议："
        "- 要求确诊"
        "- 要求开处方"
        "- 要求给出处方药具体剂量"
        "- 要求替代医生判断检查结果"
        "- 要求判断是否可以不去医院"
        "- 要求处理急危重症但不就医"
        "拒绝时不要只说不能回答，应给出安全替代方案，例如建议就医、说明需要哪些检查或建议咨询对应科室。"
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
            "4. 并非所有证据都有用\n"
            "5. 关键结论后必须紧跟引用编号，如 [1]、[2]\n"
            "6. 只能使用提供的证据编号，不能编造新的编号\n"
            "7. 如果某句话没有明确证据支撑，就不要加引用编号\n"
        )

    def generate(self, query: str, citations: Sequence[RetrievalResult]) -> GenerationResult:
        if not self.api_key or self.client is None:
            logger.warning("LLM api key is empty, returning fallback answer")
            return GenerationResult(
                query=query,
                answer="当前未配置 LLM API Key，暂无法生成答案。",
                citations=list(citations),
                retrieval_results=list(citations),
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
                retrieval_results=list(citations),
                fallback=False,
            )
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            return GenerationResult(
                query=query,
                answer="生成失败，请稍后重试或检查模型配置。",
                citations=list(citations),
                retrieval_results=list(citations),
                fallback=True,
            )


def normalize_answer_citations(
    answer: str,
    retrieval_results: Sequence[RetrievalResult],
) -> tuple[str, list[RetrievalResult]]:
    index_mapping: dict[int, int] = {}
    ordered_source_indexes: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        raw_index = int(match.group(1)) - 1
        if raw_index < 0 or raw_index >= len(retrieval_results):
            return ""
        if raw_index not in index_mapping:
            index_mapping[raw_index] = len(index_mapping) + 1
            ordered_source_indexes.append(raw_index)
        return f"[{index_mapping[raw_index]}]"

    normalized_answer = CITATION_REF_PATTERN.sub(_replace, answer)
    cited_results = [retrieval_results[index] for index in ordered_source_indexes]
    return normalized_answer, cited_results
