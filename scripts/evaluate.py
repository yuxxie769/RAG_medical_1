from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable, Literal

from openai import OpenAI

# Allow direct script execution like `uv run scripts\evaluate.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.main import AnswerRequest, AnswerResponse, answer_query, QueryRequest, query_documents
from app.config.settings import settings
from app.core.models import RetrievalResult
from app.data.cleaner import clean_records
from app.data.loader import iter_raw_records
from app.evaluation import (
    build_generation_failure_rows,
    build_generation_judge,
    run_generation_program_checks,
    summarize_generation_results,
)
from app.evaluation.judge_client import JudgeError
from app.generation import Generator, normalize_answer_citations
from app.retrieval import MilvusCollectionSchema, MilvusExportRow, MilvusStore


SearchMethod = Literal["dense", "sparse", "hybrid"]
ALL_SEARCH_METHODS: tuple[SearchMethod, ...] = ("dense", "sparse", "hybrid")
DEFAULT_REWRITE_COUNT = 3


@dataclass
class RetrievalEvalSample:
    sample_id: str
    query: str
    gold_doc_ids: list[str]
    reference_answer: str
    source: str
    source_path: str
    source_line_no: int
    rewritten_queries: list[str] = field(default_factory=list)


@dataclass
class DiscardedSample:
    query: str
    source: str
    source_path: str
    source_line_no: int
    gold_doc_ids: list[str]
    reason: str


class QueryRewriteError(RuntimeError):
    """Raised when query rewrite generation or parsing fails."""


class QueryRewriter:
    def rewrite_queries(self, query: str, rewrite_count: int) -> list[str]:
        raise NotImplementedError


class OpenAICompatibleQueryRewriter(QueryRewriter):
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: float) -> None:
        if not api_key.strip():
            raise QueryRewriteError("JUDGE_API_KEY is empty.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.timeout = timeout

    def rewrite_queries(self, query: str, rewrite_count: int) -> list[str]:
        prompt = _build_query_rewrite_prompt(query=query, rewrite_count=rewrite_count)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.timeout,
            )
        except Exception as exc:  # pragma: no cover - network/provider failures are mocked in tests
            raise QueryRewriteError(f"Rewrite request failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        return _parse_rewritten_queries_content(
            content=content,
            original_query=query,
            rewrite_count=rewrite_count,
        )


def build_store(collection_name: str) -> MilvusStore:
    schema = MilvusCollectionSchema(
        collection_name=collection_name,
        dimension=settings.embedding_dimension,
        dense_metric_type=settings.milvus_dense_metric_type,
        sparse_vector_field_name=settings.milvus_sparse_field_name,
        analyzer_type=settings.milvus_analyzer_type,
        rrf_k=settings.hybrid_rrf_k,
    )
    return MilvusStore(
        schema=schema,
        host=settings.milvus_host,
        port=settings.milvus_port,
        request_timeout_seconds=settings.milvus_request_timeout_seconds,
        management_timeout_seconds=settings.milvus_management_timeout_seconds,
    )


def build_query_rewriter() -> QueryRewriter:
    return OpenAICompatibleQueryRewriter(
        api_key=settings.judge_api_key,
        base_url=settings.judge_base_url,
        model=settings.judge_model_name,
        timeout=settings.judge_timeout_seconds,
    )


QUERY_REWRITE_SYSTEM_PROMPT = (
    "你是一个医疗检索评测集 query 改写器。"
    "你的任务是把一个中文医疗用户问题改写成多个语义等价、表达自然、适合真实检索场景的问法。"
    "不能引入新的症状、疾病、时长、人群、检查结果、治疗史或风险等级。"
    "不能改变原问题的医学含义。"
    "输出必须是 JSON。"
)


def _build_query_rewrite_prompt(*, query: str, rewrite_count: int) -> str:
    return (
        f"请将下面这个中文医疗用户问题改写成 {rewrite_count} 个不同问法。\n"
        "要求：\n"
        "1. 保持语义等价。\n"
        "2. 每条改写都要像真实用户会说的话。\n"
        "3. 不要引入原问题没有的新症状、新条件、新人群或新时间信息。\n"
        "4. 不要解释，不要编号，不要输出 Markdown。\n"
        '5. 只输出一个 JSON 对象，格式为 {"rewritten_queries": ["...", "..."]}。\n\n'
        f"原问题：{query}"
    )


def _extract_json_value(content: str) -> Any:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        if object_start >= 0 and object_end > object_start:
            try:
                return json.loads(cleaned[object_start : object_end + 1])
            except json.JSONDecodeError:
                pass
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        if array_start >= 0 and array_end > array_start:
            try:
                return json.loads(cleaned[array_start : array_end + 1])
            except json.JSONDecodeError as inner_exc:
                raise QueryRewriteError(f"Rewrite response contains malformed JSON: {inner_exc}") from exc
        raise QueryRewriteError("Rewrite response does not contain a JSON object or array.") from exc


def _normalize_rewritten_queries(
    *,
    original_query: str,
    candidates: Any,
    rewrite_count: int,
) -> list[str]:
    if not isinstance(candidates, list):
        raise QueryRewriteError("Rewrite payload must contain a list of rewritten queries.")
    normalized: list[str] = []
    seen: set[str] = set()
    original_query_normalized = original_query.strip()
    for item in candidates:
        if not isinstance(item, str):
            raise QueryRewriteError("Each rewritten query must be a string.")
        value = item.strip()
        if not value:
            raise QueryRewriteError("Rewritten queries must not be empty.")
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    if len(normalized) != rewrite_count:
        raise QueryRewriteError(
            f"Expected {rewrite_count} unique rewritten queries for {original_query_normalized!r}, "
            f"got {len(normalized)}."
        )
    return normalized


def _parse_rewritten_queries_content(
    *,
    content: str,
    original_query: str,
    rewrite_count: int,
) -> list[str]:
    payload = _extract_json_value(content)
    if isinstance(payload, dict):
        payload = payload.get("rewritten_queries")
    return _normalize_rewritten_queries(
        original_query=original_query,
        candidates=payload,
        rewrite_count=rewrite_count,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize_query_hits(hits: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in hits]


def _serialize_answer_citations(response: AnswerResponse) -> list[dict[str, Any]]:
    """Use the final answer citations for faithfulness eval."""

    return _serialize_query_hits(response.citations)


def _serialize_answer_retrieval_results(response: AnswerResponse) -> list[dict[str, Any]]:
    """Use the full retrieval hits for context relevance eval."""

    return _serialize_query_hits(response.retrieval_results)


def _load_raw_record(source_path: str, source_line_no: int) -> dict[str, Any]:
    file_path = Path(source_path)
    if source_line_no < 1:
        raise ValueError(f"Invalid source_line_no={source_line_no} for {source_path}")
    with file_path.open("r", encoding="utf-8") as file_handle:
        for line_no, line in enumerate(file_handle, start=1):
            if line_no != source_line_no:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object on line {source_line_no} in {source_path}, got {type(record).__name__}"
                )
            return {
                "raw_batch_id": file_path.stem,
                "source_path": str(file_path),
                "source_line_no": source_line_no,
                "record": record,
            }
    raise ValueError(f"Line {source_line_no} not found in {source_path}")


def _load_raw_records_batch(requests: Iterable[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped_line_numbers: dict[str, set[int]] = {}
    for source_path, source_line_no in requests:
        if source_line_no < 1:
            raise ValueError(f"Invalid source_line_no={source_line_no} for {source_path}")
        grouped_line_numbers.setdefault(source_path, set()).add(source_line_no)

    loaded_records: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path, line_numbers in grouped_line_numbers.items():
        file_path = Path(source_path)
        target_line_numbers = sorted(line_numbers)
        next_index = 0
        if not target_line_numbers:
            continue
        with file_path.open("r", encoding="utf-8") as file_handle:
            for line_no, line in enumerate(file_handle, start=1):
                while next_index < len(target_line_numbers) and target_line_numbers[next_index] < line_no:
                    next_index += 1
                if next_index >= len(target_line_numbers):
                    break
                if target_line_numbers[next_index] != line_no:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected object on line {line_no} in {source_path}, got {type(record).__name__}"
                    )
                loaded_records[(source_path, line_no)] = {
                    "raw_batch_id": file_path.stem,
                    "source_path": str(file_path),
                    "source_line_no": line_no,
                    "record": record,
                }
                next_index += 1
                if next_index >= len(target_line_numbers):
                    break

        missing_line_numbers = [line_no for line_no in target_line_numbers if (source_path, line_no) not in loaded_records]
        if missing_line_numbers:
            raise ValueError(f"Lines {missing_line_numbers} not found in {source_path}")

    return loaded_records


def resolve_reference_answer(
    *,
    question: str,
    chunk_answers: list[str],
    raw_record: dict[str, Any],
) -> str:
    cleaned_records = clean_records([raw_record], clean_batch_id="eval_recovery")
    matched_answers = [
        str(item.get("answer", "")).strip()
        for item in cleaned_records
        if str(item.get("question", "")).strip() == question.strip()
    ]
    supported_answers = []
    for answer in matched_answers:
        if answer and all(chunk_answer in answer for chunk_answer in chunk_answers):
            supported_answers.append(answer)
    unique_answers = list(dict.fromkeys(supported_answers))
    if len(unique_answers) != 1:
        source_path = str(raw_record.get("source_path", ""))
        source_line_no = int(raw_record.get("source_line_no") or 0)
        raise ValueError(
            f"Expected exactly one full answer for question={question!r} at {source_path}:{source_line_no}, "
            f"got {len(unique_answers)}"
        )
    return unique_answers[0]


def build_retrieval_samples(
    rows: Iterable[MilvusExportRow],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[RetrievalEvalSample], list[DiscardedSample]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.source_path, row.source_line_no, row.question.strip())
        group = grouped.setdefault(
            key,
            {
                "query": row.question.strip(),
                "source": row.source,
                "source_path": row.source_path,
                "source_line_no": row.source_line_no,
                "gold_doc_ids": [],
                "chunk_answers": [],
            },
        )
        if row.doc_id not in group["gold_doc_ids"]:
            group["gold_doc_ids"].append(row.doc_id)
        normalized_chunk_answer = row.answer.strip()
        if normalized_chunk_answer and normalized_chunk_answer not in group["chunk_answers"]:
            group["chunk_answers"].append(normalized_chunk_answer)

    raw_records = _load_raw_records_batch((source_path, source_line_no) for source_path, source_line_no, _ in grouped)

    successful: list[RetrievalEvalSample] = []
    discarded: list[DiscardedSample] = []
    for source_path, source_line_no, query in grouped:
        group = grouped[(source_path, source_line_no, query)]
        try:
            reference_answer = resolve_reference_answer(
                question=group["query"],
                chunk_answers=group["chunk_answers"],
                raw_record=raw_records[(group["source_path"], group["source_line_no"])],
            )
        except Exception as exc:
            discarded.append(
                DiscardedSample(
                    query=group["query"],
                    source=group["source"],
                    source_path=group["source_path"],
                    source_line_no=group["source_line_no"],
                    gold_doc_ids=list(group["gold_doc_ids"]),
                    reason=str(exc),
                )
            )
            continue
        successful.append(
            RetrievalEvalSample(
                sample_id="",
                query=group["query"],
                gold_doc_ids=list(group["gold_doc_ids"]),
                reference_answer=reference_answer,
                source=group["source"],
                source_path=group["source_path"],
                source_line_no=group["source_line_no"],
            )
        )

    rng = random.Random(seed)
    if sample_size < len(successful):
        successful = rng.sample(successful, sample_size)
    for index, sample in enumerate(successful, start=1):
        sample.sample_id = f"eval_{index:06d}"
    return successful, discarded


def _attach_rewritten_queries(
    samples: list[RetrievalEvalSample],
    *,
    rewrite_count: int,
) -> None:
    if not samples:
        return
    rewriter = build_query_rewriter()
    for sample in samples:
        sample.rewritten_queries = _normalize_rewritten_queries(
            original_query=sample.query,
            candidates=rewriter.rewrite_queries(sample.query, rewrite_count),
            rewrite_count=rewrite_count,
        )


def build_retrieval_set(
    *,
    collection: str,
    sample_size: int,
    seed: int,
    output_path: Path,
    rewrite_count: int = DEFAULT_REWRITE_COUNT,
) -> dict[str, Any]:
    store = build_store(collection)
    target_group_count = max(sample_size * 3, sample_size)
    candidate_group_keys: list[tuple[str, int, str]] = []
    seen_group_keys: set[tuple[str, int, str]] = set()
    scanned_rows = 0
    for row in store.iter_export_rows():
        scanned_rows += 1
        key = (row.source_path, row.source_line_no, row.question.strip())
        if key in seen_group_keys:
            continue
        seen_group_keys.add(key)
        candidate_group_keys.append(key)
        if len(candidate_group_keys) >= target_group_count:
            break

    exported_rows: list[MilvusExportRow] = []
    for source_path, source_line_no, question in candidate_group_keys:
        exported_rows.extend(
            store.export_rows_for_group(
                source_path=source_path,
                source_line_no=source_line_no,
                question=question,
            )
        )

    samples, discarded = build_retrieval_samples(exported_rows, sample_size=sample_size, seed=seed)
    _attach_rewritten_queries(samples, rewrite_count=rewrite_count)
    _write_jsonl(output_path, [asdict(sample) for sample in samples])
    discarded_path = output_path.with_name(f"{output_path.stem}.discarded.jsonl")
    _write_jsonl(discarded_path, [asdict(item) for item in discarded])
    return {
        "collection": collection,
        "candidate_count": len(exported_rows),
        "scanned_rows": scanned_rows,
        "candidate_group_count": len(candidate_group_keys),
        "sample_count": len(samples),
        "discarded_count": len(discarded),
        "rewrite_count": rewrite_count,
        "output_path": str(output_path),
        "discarded_path": str(discarded_path),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * percentile))))
    return float(sorted_values[index])


def _resolve_search_methods(search_method: str) -> list[SearchMethod]:
    if search_method == "all":
        return list(ALL_SEARCH_METHODS)
    return [search_method]  # type: ignore[list-item]


def _validate_retrieval_samples_for_rewrites(
    samples: list[RetrievalEvalSample],
    *,
    rewrite_count: int = DEFAULT_REWRITE_COUNT,
) -> None:
    for sample in samples:
        try:
            sample.rewritten_queries = _normalize_rewritten_queries(
                original_query=sample.query,
                candidates=sample.rewritten_queries,
                rewrite_count=rewrite_count,
            )
        except QueryRewriteError as exc:
            raise ValueError(
                f"Dataset sample {sample.sample_id or '<unknown>'} is missing valid rewritten_queries. "
                f"Rebuild the dataset with build_retrieval_set. Details: {exc}"
            ) from exc


def run_retrieval_evaluation(
    *,
    dataset_path: Path,
    search_method: str,
    top_k: int,
    fetch_k: int,
    output_dir: Path,
) -> dict[str, Any]:
    samples = [RetrievalEvalSample(**row) for row in _read_jsonl(dataset_path)]
    _validate_retrieval_samples_for_rewrites(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_query_count = len(samples) * DEFAULT_REWRITE_COUNT
    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "fetch_k": fetch_k,
        "sample_count": len(samples),
        "evaluation_query_count": evaluation_query_count,
        "rewrite_count_per_sample": DEFAULT_REWRITE_COUNT,
        "rewritten_query_mode": "dataset_precomputed",
        "methods": {},
    }
    failures_path = output_dir / "retrieval_failures.jsonl"
    all_failures: list[dict[str, Any]] = []

    for method in _resolve_search_methods(search_method):
        recall_at_5_total = 0.0
        recall_at_10_total = 0.0
        mrr_total = 0.0
        latencies: list[float] = []

        for sample in samples:
            for rewrite_index, rewritten_query in enumerate(sample.rewritten_queries, start=1):
                started = perf_counter()
                response = query_documents(
                    QueryRequest(
                        query=rewritten_query,
                        search_method=method,
                        top_k=top_k,
                        fetch_k=fetch_k,
                    )
                )
                latency = perf_counter() - started
                latencies.append(latency)

                hit_doc_ids = [hit.doc_id for hit in response.hits]
                first_five = hit_doc_ids[:5]
                first_ten = hit_doc_ids[:10]
                gold_doc_ids = set(sample.gold_doc_ids)
                recall_at_5_hit = any(doc_id in gold_doc_ids for doc_id in first_five)
                recall_at_10_hit = any(doc_id in gold_doc_ids for doc_id in first_ten)
                recall_at_5_total += float(recall_at_5_hit)
                recall_at_10_total += float(recall_at_10_hit)

                reciprocal_rank = 0.0
                for rank, doc_id in enumerate(first_ten, start=1):
                    if doc_id in gold_doc_ids:
                        reciprocal_rank = 1.0 / rank
                        break
                mrr_total += reciprocal_rank

                if not recall_at_10_hit:
                    all_failures.append(
                        {
                            "search_method": method,
                            "sample_id": sample.sample_id,
                            "original_query": sample.query,
                            "rewritten_query": rewritten_query,
                            "rewrite_index": rewrite_index,
                            "gold_doc_ids": sample.gold_doc_ids,
                            "returned_doc_ids": hit_doc_ids,
                            "source_path": sample.source_path,
                            "source_line_no": sample.source_line_no,
                        }
                    )

        sample_count = len(samples)
        recall_at_5 = recall_at_5_total / evaluation_query_count if evaluation_query_count else 0.0
        recall_at_10 = recall_at_10_total / evaluation_query_count if evaluation_query_count else 0.0
        mrr = mrr_total / evaluation_query_count if evaluation_query_count else 0.0
        summary["methods"][method] = {
            "sample_count": sample_count,
            "evaluation_query_count": evaluation_query_count,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "mrr": mrr,
            "latency_p50": median(latencies) if latencies else 0.0,
            "latency_p95": _percentile(latencies, 0.95),
            "meets_recall_at_10_target": recall_at_10 >= 0.75,
        }

    summary_path = output_dir / "retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(failures_path, all_failures)
    return {
        "summary_path": str(summary_path),
        "failures_path": str(failures_path),
        "method_count": len(summary["methods"]),
    }


def run_generation_evaluation(
    *,
    dataset_path: Path,
    output_dir: Path,
    top_k: int,
    sample_limit: int | None,
) -> dict[str, Any]:
    samples = [RetrievalEvalSample(**row) for row in _read_jsonl(dataset_path)]
    if sample_limit is not None:
        samples = samples[:sample_limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        judge = build_generation_judge()
        judge_boot_error: str | None = None
    except Exception as exc:
        judge = None
        judge_boot_error = str(exc)

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[generation] sample {index}/{len(samples)} start: {sample.sample_id} {sample.query}")

        print(f"[generation] sample {index}/{len(samples)} answering...")
        started = perf_counter()
        response = answer_query(AnswerRequest(query=sample.query, top_k=top_k))
        latency_ms = (perf_counter() - started) * 1000
        print(f"[generation] sample {index}/{len(samples)} answer done in {latency_ms:.2f} ms")
        answer_citations = _serialize_answer_citations(response)
        answer_retrieval_results = _serialize_answer_retrieval_results(response)
        print(f"[generation] sample {index}/{len(samples)} program checks...")
        program_checks = run_generation_program_checks(
            query=sample.query,
            answer=response.answer,
            citations=answer_citations,
            fallback=response.fallback,
            latency_ms=latency_ms,
            latency_target_seconds=settings.generation_timeout_seconds,
        )

        llm_judge_result: dict[str, Any] | None = None
        judge_error = False
        judge_error_reason: str | None = None
        if judge is None:
            judge_error = True
            judge_error_reason = judge_boot_error or "Judge is unavailable."
        else:
            try:
                llm_judge_result = judge.evaluate_response(
                    query=sample.query,
                    answer=response.answer,
                    citations=answer_citations,
                    retrieval_results=answer_retrieval_results,
                    reference_answer=sample.reference_answer,
                )
            except JudgeError as exc:
                judge_error = True
                judge_error_reason = str(exc)

        results.append(
            {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "reference_answer": sample.reference_answer,
                "source": sample.source,
                "gold_doc_ids": sample.gold_doc_ids,
                "answer": response.answer,
                "fallback": response.fallback,
                "citations": answer_citations,
                "retrieval_results": answer_retrieval_results,
                "latency_ms": latency_ms,
                "program_checks": program_checks,
                "llm_judge_result": llm_judge_result,
                "judge_error": judge_error,
                "judge_error_reason": judge_error_reason,
            }
        )

    summary = summarize_generation_results(results)
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "top_k": top_k,
            "notes": [
                "medical_correctness is treated as a lower-confidence auxiliary dimension in v1.",
            ],
        }
    )

    results_path = output_dir / "generation_results.jsonl"
    summary_path = output_dir / "generation_summary.json"
    failures_path = output_dir / "generation_failures.jsonl"
    _write_jsonl(results_path, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(failures_path, build_generation_failure_rows(results))
    return {
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "failures_path": str(failures_path),
        "sample_count": len(results),
        "judge_error_count": summary["judge_error_count"],
    }


def _build_source_generation_samples(
    *,
    source_path: Path,
    sample_limit: int | None,
) -> list[RetrievalEvalSample]:
    samples: list[RetrievalEvalSample] = []
    sample_index = 0
    for raw_record in iter_raw_records(str(source_path), batch_id=source_path.stem):
        cleaned_records = clean_records([raw_record], clean_batch_id="generation_source_eval")
        for item in cleaned_records:
            sample_index += 1
            samples.append(
                RetrievalEvalSample(
                    sample_id=f"source_eval_{sample_index:06d}",
                    query=str(item.get("question", "")).strip(),
                    gold_doc_ids=[],
                    reference_answer=str(item.get("answer", "")).strip(),
                    source=str(source_path),
                    source_path=str(source_path),
                    source_line_no=int(item.get("source_line_no") or 0),
                )
            )
            if sample_limit is not None and len(samples) >= sample_limit:
                return samples
    return samples


def _build_oracle_retrieval_results(sample: RetrievalEvalSample) -> list[RetrievalResult]:
    content = f"Q: {sample.query}\nA: {sample.reference_answer}"
    doc_id = hashlib.sha256(
        f"{sample.query}|{sample.source_path}|{sample.source_line_no}".encode("utf-8")
    ).hexdigest()
    return [
        RetrievalResult(
            doc_id=doc_id,
            content=content,
            score=1.0,
            metadata={
                "source_path": sample.source_path,
                "source_line_no": sample.source_line_no,
                "oracle_evidence": True,
            },
        )
    ]


def run_source_generation_evaluation(
    *,
    source_path: Path,
    output_dir: Path,
    sample_limit: int | None,
) -> dict[str, Any]:
    samples = _build_source_generation_samples(source_path=source_path, sample_limit=sample_limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model_name,
        timeout=settings.llm_timeout_seconds,
    )

    try:
        judge = build_generation_judge()
        judge_boot_error: str | None = None
    except Exception as exc:
        judge = None
        judge_boot_error = str(exc)

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[source-generation] sample {index}/{len(samples)} start: {sample.sample_id} {sample.query}")

        retrieval_results = _build_oracle_retrieval_results(sample)
        print(f"[source-generation] sample {index}/{len(samples)} generating with oracle evidence...")
        started = perf_counter()
        generation_result = generator.generate(sample.query, retrieval_results)
        normalized_answer, cited_results = normalize_answer_citations(generation_result.answer, retrieval_results)
        latency_ms = (perf_counter() - started) * 1000
        print(f"[source-generation] sample {index}/{len(samples)} answer done in {latency_ms:.2f} ms")

        response = AnswerResponse(
            query=sample.query,
            answer=normalized_answer,
            fallback=generation_result.fallback,
            citations=cited_results,
            retrieval_results=retrieval_results,
        )
        answer_citations = _serialize_answer_citations(response)
        answer_retrieval_results = _serialize_answer_retrieval_results(response)

        print(f"[source-generation] sample {index}/{len(samples)} program checks...")
        program_checks = run_generation_program_checks(
            query=sample.query,
            answer=response.answer,
            citations=answer_citations,
            fallback=response.fallback,
            latency_ms=latency_ms,
            latency_target_seconds=settings.generation_timeout_seconds,
        )

        llm_judge_result: dict[str, Any] | None = None
        judge_error = False
        judge_error_reason: str | None = None
        if judge is None:
            judge_error = True
            judge_error_reason = judge_boot_error or "Judge is unavailable."
        else:
            try:
                llm_judge_result = judge.evaluate_response(
                    query=sample.query,
                    answer=response.answer,
                    citations=answer_citations,
                    retrieval_results=answer_retrieval_results,
                    reference_answer=sample.reference_answer,
                )
            except JudgeError as exc:
                judge_error = True
                judge_error_reason = str(exc)

        results.append(
            {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "reference_answer": sample.reference_answer,
                "source": sample.source,
                "gold_doc_ids": sample.gold_doc_ids,
                "answer": response.answer,
                "fallback": response.fallback,
                "citations": answer_citations,
                "retrieval_results": answer_retrieval_results,
                "latency_ms": latency_ms,
                "program_checks": program_checks,
                "llm_judge_result": llm_judge_result,
                "judge_error": judge_error,
                "judge_error_reason": judge_error_reason,
            }
        )

    summary = summarize_generation_results(results)
    summary.update(
        {
            "dataset_path": str(source_path),
            "top_k": 1,
            "evaluation_mode": "oracle_source_generation",
            "notes": [
                "This mode bypasses Milvus retrieval and feeds each source sample's own answer as oracle evidence.",
                "Use it to isolate generation quality from retrieval quality on data that has not been ingested.",
                "medical_correctness is treated as a lower-confidence auxiliary dimension in v1.",
            ],
        }
    )

    results_path = output_dir / "generation_results.jsonl"
    summary_path = output_dir / "generation_summary.json"
    failures_path = output_dir / "generation_failures.jsonl"
    _write_jsonl(results_path, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(failures_path, build_generation_failure_rows(results))
    return {
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "failures_path": str(failures_path),
        "sample_count": len(results),
        "judge_error_count": summary["judge_error_count"],
        "evaluation_mode": "oracle_source_generation",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation utilities for stage 3 retrieval baselines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build_retrieval_set")
    build_parser.add_argument("--collection", required=True)
    build_parser.add_argument("--sample-size", type=int, default=300)
    build_parser.add_argument("--seed", type=int, default=20260608)
    build_parser.add_argument("--output", default="eval/datasets/retrieval_eval.jsonl")
    build_parser.add_argument("--rewrite-count", type=int, default=DEFAULT_REWRITE_COUNT)

    retrieval_parser = subparsers.add_parser("retrieval")
    retrieval_parser.add_argument("--dataset", default="eval/datasets/retrieval_eval.jsonl")
    retrieval_parser.add_argument("--search-method", choices=[*ALL_SEARCH_METHODS, "all"], default="all")
    retrieval_parser.add_argument("--top-k", type=int, default=10)
    retrieval_parser.add_argument("--fetch-k", type=int, default=max(10, settings.fetch_k))
    retrieval_parser.add_argument("--output-dir", default="eval/reports")

    generation_parser = subparsers.add_parser("generation")
    generation_parser.add_argument("--dataset", default="eval/datasets/retrieval_eval.jsonl")
    generation_parser.add_argument("--output-dir", default="eval/reports")
    generation_parser.add_argument("--top-k", type=int, default=settings.top_k)
    generation_parser.add_argument("--sample-limit", type=int, default=None)

    source_generation_parser = subparsers.add_parser("source_generation")
    source_generation_parser.add_argument("--source-path", required=True)
    source_generation_parser.add_argument("--output-dir", default="eval/reports")
    source_generation_parser.add_argument("--sample-limit", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build_retrieval_set":
        result = build_retrieval_set(
            collection=args.collection,
            sample_size=args.sample_size,
            seed=args.seed,
            output_path=Path(args.output),
            rewrite_count=args.rewrite_count,
        )
    elif args.command == "retrieval":
        result = run_retrieval_evaluation(
            dataset_path=Path(args.dataset),
            search_method=args.search_method,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            output_dir=Path(args.output_dir),
        )
    elif args.command == "generation":
        result = run_generation_evaluation(
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            top_k=args.top_k,
            sample_limit=args.sample_limit,
        )
    elif args.command == "source_generation":
        result = run_source_generation_evaluation(
            source_path=Path(args.source_path),
            output_dir=Path(args.output_dir),
            sample_limit=args.sample_limit,
        )
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
