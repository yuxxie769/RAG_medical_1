from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List
from uuid import uuid4

from app.core.models import Document


ANSWER_CHUNK_THRESHOLD = 1648
ANSWER_CHUNK_SIZE = 700
ANSWER_CHUNK_OVERLAP = 70
ANSWER_CHUNK_STEP = ANSWER_CHUNK_SIZE - ANSWER_CHUNK_OVERLAP


def _build_content(question: str, answer: str) -> str:
    return f"Q: {question}\nA: {answer}"


def _split_answer(answer: str) -> List[tuple[str, int, int]]:
    """Split only long answers into overlapping character windows."""
    if len(answer) <= ANSWER_CHUNK_THRESHOLD:
        return [(answer, 0, len(answer))]

    chunks: List[tuple[str, int, int]] = []
    for start in range(0, len(answer), ANSWER_CHUNK_STEP):
        end = min(start + ANSWER_CHUNK_SIZE, len(answer))
        chunks.append((answer[start:end], start, end))
        if end == len(answer):
            break
    return chunks


def _build_stable_doc_id(question: str, answer: str, metadata: Dict[str, Any]) -> str:
    """Build a deterministic document id from stable content and source metadata."""
    raw = "\n".join(
        [
            question,
            answer,
            str(metadata.get("source_path", "unknown")),
            str(metadata.get("source_line_no") or ""),
        ]
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _build_chunk_doc_id(parent_doc_id: str, chunk_index: int) -> str:
    return sha256(f"{parent_doc_id}\n{chunk_index}".encode("utf-8")).hexdigest()


def build_documents(records: List[Dict[str, Any]], batch_id: str | None = None) -> List[Document]:
    """Build knowledge base documents from cleaned records."""
    resolved_batch_id = batch_id or f"kb_{uuid4().hex[:8]}"
    documents: List[Document] = []

    for index, item in enumerate(records):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        base_metadata = dict(item.get("meta", {}))
        base_metadata.update(
            {
                "source": item.get("source_path", "unknown"),
                "split": str(item.get("meta", {}).get("split", item.get("split", "unknown"))),
                "clean_batch_id": item.get("clean_batch_id", "unknown"),
                "raw_batch_id": item.get("raw_batch_id", "unknown"),
                "source_path": item.get("source_path", "unknown"),
                "source_line_no": item.get("source_line_no"),
                "kb_batch_id": resolved_batch_id,
                "kb_index": index,
            }
        )
        parent_doc_id = _build_stable_doc_id(question, answer, base_metadata)
        answer_chunks = _split_answer(answer)
        is_chunked = len(answer_chunks) > 1

        for chunk_index, (chunk_answer, chunk_start, chunk_end) in enumerate(answer_chunks):
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "is_chunked": is_chunked,
                    "parent_doc_id": parent_doc_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(answer_chunks),
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "original_answer_length": len(answer),
                }
            )
            doc_id = _build_chunk_doc_id(parent_doc_id, chunk_index) if is_chunked else parent_doc_id
            documents.append(
                Document(
                    doc_id=doc_id,
                    question=question,
                    answer=chunk_answer,
                    content=_build_content(question, chunk_answer),
                    metadata=metadata,
                )
            )

    return documents
