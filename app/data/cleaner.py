from __future__ import annotations

from typing import Any, Dict, List


def _normalize_text_items(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        flattened: List[str] = []
        for item in value:
            flattened.extend(_normalize_text_items(item))
        return flattened
    return []


def _extract_questions(record: Dict[str, Any]) -> List[str]:
    questions = _normalize_text_items(record.get("question"))
    if questions:
        return questions
    return _normalize_text_items(record.get("questions"))


def _extract_answers(record: Dict[str, Any]) -> List[str]:
    answers = _normalize_text_items(record.get("answer"))
    if answers:
        return answers
    return _normalize_text_items(record.get("answers"))


def clean_records(records: List[Dict[str, Any]], clean_batch_id: str | None = None) -> List[Dict[str, Any]]:
    """Clean raw records into a normalized list.

    Rules:
    - flatten nested question/answer lists
    - support both singular and plural field names
    - keep non-empty question/answer pairs
    - deduplicate by question+answer
    - preserve source metadata for downstream modules
    """
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    resolved_clean_batch_id = clean_batch_id or "clean_batch"

    for index, item in enumerate(records):
        source_path = item.get("source_path", "unknown")
        source_line_no = item.get("source_line_no")
        raw_batch_id = item.get("raw_batch_id", "unknown")
        record = item.get("record", item)

        raw_questions = _extract_questions(record)
        raw_answers = _extract_answers(record)
        if not raw_questions or not raw_answers:
            continue

        question_type = "nested" if isinstance(record.get("questions"), list) and any(isinstance(q, list) for q in record.get("questions", [])) else "single"

        if len(raw_answers) == 1:
            pairs = [(question, raw_answers[0]) for question in raw_questions]
        elif len(raw_questions) == 1:
            pairs = [(raw_questions[0], answer) for answer in raw_answers]
        else:
            pairs = [(question, answer) for question in raw_questions for answer in raw_answers]

        for question, answer in pairs:
            key = (question, answer)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(
                {
                    "clean_batch_id": resolved_clean_batch_id,
                    "source_path": source_path,
                    "source_line_no": source_line_no,
                    "raw_batch_id": raw_batch_id,
                    "question": question,
                    "answer": answer,
                    "meta": {
                        "raw_index": index,
                        "question_type": question_type,
                    },
                }
            )

    return cleaned
