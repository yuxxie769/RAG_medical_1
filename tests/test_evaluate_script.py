import json

from app.api.main import QueryHit, QueryResponse
from app.data.cleaner import clean_records
from app.kb.builder import build_documents
from app.retrieval import InMemoryMilvusStore, MilvusCollectionSchema, build_records
from scripts import evaluate


def test_iter_export_rows_reads_expected_fields():
    store = InMemoryMilvusStore(MilvusCollectionSchema(collection_name="eval_test", dimension=2))
    cleaned = clean_records(
        [
            {
                "raw_batch_id": "raw_eval",
                "source_path": "sample.jsonl",
                "source_line_no": 1,
                "record": {"question": "Q", "answer": "A"},
            }
        ],
        clean_batch_id="clean_eval",
    )
    documents = build_documents(cleaned, batch_id="kb_eval")
    store.upsert(build_records(documents, [[0.1, 0.2]]))

    rows = list(store.iter_export_rows())

    assert len(rows) == 1
    assert rows[0].question == "Q"
    assert rows[0].answer == "A"
    assert rows[0].source_path == "sample.jsonl"
    assert rows[0].source_line_no == 1


def test_build_retrieval_samples_groups_chunked_doc_ids_and_recovers_full_answer(tmp_path):
    source_path = tmp_path / "chunked.jsonl"
    long_answer = "A" * 1700
    source_path.write_text(
        json.dumps({"question": "Q", "answer": long_answer}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cleaned = clean_records(
        [
            {
                "raw_batch_id": source_path.stem,
                "source_path": str(source_path),
                "source_line_no": 1,
                "record": {"question": "Q", "answer": long_answer},
            }
        ],
        clean_batch_id="clean_eval",
    )
    documents = build_documents(cleaned, batch_id="kb_eval")
    rows = [
        evaluate.MilvusExportRow(
            doc_id=doc.doc_id,
            question=doc.question,
            answer=doc.answer,
            source=str(doc.metadata.get("source", "")),
            source_path=str(doc.metadata.get("source_path", "")),
            source_line_no=int(doc.metadata.get("source_line_no") or 0),
        )
        for doc in documents
    ]

    samples, discarded = evaluate.build_retrieval_samples(rows, sample_size=10, seed=1)

    assert discarded == []
    assert len(samples) == 1
    assert samples[0].reference_answer == long_answer
    assert len(samples[0].gold_doc_ids) == len(documents)


def test_build_retrieval_samples_discards_ambiguous_reference_answers(tmp_path):
    source_path = tmp_path / "ambiguous.jsonl"
    source_path.write_text(
        json.dumps({"question": "Q", "answers": ["abc one", "abc two"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        evaluate.MilvusExportRow(
            doc_id="doc_1",
            question="Q",
            answer="abc",
            source=str(source_path),
            source_path=str(source_path),
            source_line_no=1,
        )
    ]

    samples, discarded = evaluate.build_retrieval_samples(rows, sample_size=10, seed=1)

    assert samples == []
    assert len(discarded) == 1
    assert "Expected exactly one full answer" in discarded[0].reason


def test_build_retrieval_set_writes_dataset_and_discard_logs(tmp_path, monkeypatch):
    source_path = tmp_path / "samples.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Q1", "answer": "完整答案1"}, ensure_ascii=False),
                json.dumps({"question": "Q2", "answers": ["abc one", "abc two"]}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        evaluate.MilvusExportRow(
            doc_id="doc_1",
            question="Q1",
            answer="完整答案1",
            source=str(source_path),
            source_path=str(source_path),
            source_line_no=1,
        ),
        evaluate.MilvusExportRow(
            doc_id="doc_2",
            question="Q2",
            answer="abc",
            source=str(source_path),
            source_path=str(source_path),
            source_line_no=2,
        ),
    ]

    class FakeStore:
        def iter_export_rows(self):
            return iter(rows)

        def export_rows_for_group(self, *, source_path, source_line_no, question):
            return [
                row
                for row in rows
                if row.source_path == source_path and row.source_line_no == source_line_no and row.question == question
            ]

    monkeypatch.setattr(evaluate, "build_store", lambda collection: FakeStore())

    output_path = tmp_path / "retrieval_eval.jsonl"
    result = evaluate.build_retrieval_set(
        collection="test_collection",
        sample_size=10,
        seed=1,
        output_path=output_path,
    )

    dataset_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    discarded_path = output_path.with_name("retrieval_eval.discarded.jsonl")
    discarded_rows = [json.loads(line) for line in discarded_path.read_text(encoding="utf-8").splitlines()]

    assert result["sample_count"] == 1
    assert result["discarded_count"] == 1
    assert result["candidate_group_count"] == 2
    assert dataset_rows[0]["query"] == "Q1"
    assert discarded_rows[0]["query"] == "Q2"


def test_run_retrieval_evaluation_supports_multiple_gold_doc_ids(tmp_path, monkeypatch):
    dataset_path = tmp_path / "retrieval_eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "sample_id": "eval_000001",
                "query": "Q",
                "gold_doc_ids": ["doc_2", "doc_3"],
                "reference_answer": "A",
                "source": "sample",
                "source_path": "sample.jsonl",
                "source_line_no": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_query_documents(request):
        hits_by_method = {
            "dense": [
                QueryHit(doc_id="doc_x", content="x", score=0.9, metadata={}),
                QueryHit(doc_id="doc_3", content="y", score=0.8, metadata={}),
            ],
            "sparse": [
                QueryHit(doc_id="doc_none", content="z", score=0.7, metadata={}),
            ],
            "hybrid": [
                QueryHit(doc_id="doc_2", content="h", score=0.95, metadata={}),
            ],
        }
        return QueryResponse(
            query=request.query,
            search_method=request.search_method,
            top_k=request.top_k,
            fetch_k=request.fetch_k or request.top_k,
            min_score=0.0,
            hits=hits_by_method[request.search_method],
        )

    monkeypatch.setattr(evaluate, "query_documents", fake_query_documents)

    output_dir = tmp_path / "reports"
    result = evaluate.run_retrieval_evaluation(
        dataset_path=dataset_path,
        search_method="all",
        top_k=10,
        fetch_k=10,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "retrieval_summary.json").read_text(encoding="utf-8"))
    failures = [json.loads(line) for line in (output_dir / "retrieval_failures.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result["method_count"] == 3
    assert summary["methods"]["dense"]["recall_at_10"] == 1.0
    assert summary["methods"]["dense"]["mrr"] == 0.5
    assert summary["methods"]["sparse"]["recall_at_10"] == 0.0
    assert summary["methods"]["hybrid"]["recall_at_10"] == 1.0
    assert len(failures) == 1
    assert failures[0]["search_method"] == "sparse"
