from app.kb import build_documents
from app.kb.builder import ANSWER_CHUNK_OVERLAP, ANSWER_CHUNK_THRESHOLD


def test_build_documents_generates_stable_unique_doc_ids():
    records = [
        {
            "clean_batch_id": "clean_001",
            "raw_batch_id": "raw_001",
            "source_path": "sample.jsonl",
            "source_line_no": 1,
            "question": "A",
            "answer": "1",
            "meta": {"question_type": "single"},
        },
        {
            "clean_batch_id": "clean_001",
            "raw_batch_id": "raw_001",
            "source_path": "sample.jsonl",
            "source_line_no": 2,
            "question": "B",
            "answer": "2",
            "meta": {"question_type": "single"},
        },
    ]

    docs = build_documents(records, batch_id="kb_001")
    repeated_docs = build_documents(records, batch_id="kb_002")

    assert len(docs) == 2
    assert docs[0].doc_id == repeated_docs[0].doc_id
    assert docs[1].doc_id == repeated_docs[1].doc_id
    assert docs[0].doc_id != docs[1].doc_id
    assert docs[0].content == "Q: A\nA: 1"
    assert docs[0].metadata["clean_batch_id"] == "clean_001"
    assert docs[0].metadata["raw_batch_id"] == "raw_001"
    assert docs[0].metadata["source"] == "sample.jsonl"
    assert docs[0].metadata["split"] == "unknown"
    assert docs[0].metadata["kb_batch_id"] == "kb_001"
    assert docs[0].metadata["is_chunked"] is False
    assert docs[0].metadata["parent_doc_id"] == docs[0].doc_id
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[0].metadata["chunk_count"] == 1
    assert docs[0].metadata["chunk_start"] == 0
    assert docs[0].metadata["chunk_end"] == 1
    assert docs[0].metadata["original_answer_length"] == 1


def test_build_documents_skips_invalid_records():
    records = [
        {"question": "", "answer": "1"},
        {"question": "A", "answer": ""},
        {"question": "A", "answer": "1"},
    ]

    docs = build_documents(records, batch_id="kb_002")

    assert len(docs) == 1
    assert len(docs[0].doc_id) == 64


def test_build_documents_chunks_only_answers_above_threshold():
    records = [
        {
            "source_path": "sample.jsonl",
            "source_line_no": 1,
            "question": "Q",
            "answer": "a" * ANSWER_CHUNK_THRESHOLD,
        },
        {
            "source_path": "sample.jsonl",
            "source_line_no": 2,
            "question": "Q",
            "answer": "b" * (ANSWER_CHUNK_THRESHOLD + 1),
        },
    ]

    docs = build_documents(records, batch_id="kb_chunks")
    unchunked = docs[0]
    chunks = docs[1:]

    assert unchunked.metadata["is_chunked"] is False
    assert len(chunks) == 3
    assert [len(doc.answer) for doc in chunks] == [700, 700, 389]
    assert [doc.metadata["chunk_start"] for doc in chunks] == [0, 630, 1260]
    assert [doc.metadata["chunk_end"] for doc in chunks] == [700, 1330, 1649]
    assert all(doc.metadata["is_chunked"] is True for doc in chunks)
    assert all(doc.metadata["chunk_count"] == 3 for doc in chunks)
    assert len({doc.doc_id for doc in chunks}) == 3
    assert len({doc.metadata["parent_doc_id"] for doc in chunks}) == 1
    assert chunks[0].answer[-ANSWER_CHUNK_OVERLAP:] == chunks[1].answer[:ANSWER_CHUNK_OVERLAP]
    assert all(doc.content == f"Q: Q\nA: {doc.answer}" for doc in chunks)
