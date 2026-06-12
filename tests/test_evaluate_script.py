import json

import pytest

from app.api.main import AnswerResponse, QueryHit, QueryResponse
from app.data.cleaner import clean_records
from app.kb.builder import build_documents
from app.retrieval import InMemoryMilvusStore, MilvusCollectionSchema, build_records
from scripts import evaluate


class StaticRewriter:
    def __init__(self, rewrites_by_query):
        self.rewrites_by_query = rewrites_by_query

    def rewrite_queries(self, query, rewrite_count):
        return self.rewrites_by_query[query]


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
    monkeypatch.setattr(
        evaluate,
        "build_query_rewriter",
        lambda: StaticRewriter({"Q1": ["Q1改写1", "Q1改写2", "Q1改写3"]}),
    )

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
    assert result["rewrite_count"] == 3
    assert dataset_rows[0]["query"] == "Q1"
    assert dataset_rows[0]["rewritten_queries"] == ["Q1改写1", "Q1改写2", "Q1改写3"]
    assert discarded_rows[0]["query"] == "Q2"


@pytest.mark.parametrize(
    "bad_rewrites, expected_message",
    [
        (["重复问法", "重复问法", "另一个问法"], "Expected 3 unique rewritten queries"),
        (["", "问法2", "问法3"], "Rewritten queries must not be empty"),
        (["只有一个", "只有两个"], "Expected 3 unique rewritten queries"),
    ],
)
def test_build_retrieval_set_fails_when_rewritten_queries_invalid(tmp_path, monkeypatch, bad_rewrites, expected_message):
    source_path = tmp_path / "samples.jsonl"
    source_path.write_text(
        json.dumps({"question": "Q1", "answer": "完整答案1"}, ensure_ascii=False) + "\n",
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
        )
    ]

    class FakeStore:
        def iter_export_rows(self):
            return iter(rows)

        def export_rows_for_group(self, *, source_path, source_line_no, question):
            return list(rows)

    monkeypatch.setattr(evaluate, "build_store", lambda collection: FakeStore())
    monkeypatch.setattr(
        evaluate,
        "build_query_rewriter",
        lambda: StaticRewriter({"Q1": bad_rewrites}),
    )

    with pytest.raises(evaluate.QueryRewriteError, match=expected_message):
        evaluate.build_retrieval_set(
            collection="test_collection",
            sample_size=10,
            seed=1,
            output_path=tmp_path / "retrieval_eval.jsonl",
        )


def test_parse_rewritten_queries_content_rejects_non_structured_response():
    with pytest.raises(evaluate.QueryRewriteError, match="does not contain a JSON object or array"):
        evaluate._parse_rewritten_queries_content(
            content="这不是结构化返回",
            original_query="Q1",
            rewrite_count=3,
        )


def test_run_retrieval_evaluation_supports_multiple_gold_doc_ids(tmp_path, monkeypatch):
    dataset_path = tmp_path / "retrieval_eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "sample_id": "eval_000001",
                "query": "Q",
                "rewritten_queries": ["Q改写1", "Q改写2", "Q改写3"],
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

    seen_queries = []

    def fake_query_documents(request):
        seen_queries.append((request.search_method, request.query))
        hits_by_method_and_query = {
            ("dense", "Q改写1"): [QueryHit(doc_id="doc_x", content="x", score=0.9, metadata={})],
            ("dense", "Q改写2"): [
                QueryHit(doc_id="doc_x", content="x", score=0.9, metadata={}),
                QueryHit(doc_id="doc_3", content="y", score=0.8, metadata={}),
            ],
            ("dense", "Q改写3"): [QueryHit(doc_id="doc_2", content="z", score=0.7, metadata={})],
            ("sparse", "Q改写1"): [QueryHit(doc_id="doc_none", content="z", score=0.7, metadata={})],
            ("sparse", "Q改写2"): [QueryHit(doc_id="doc_2", content="h", score=0.95, metadata={})],
            ("sparse", "Q改写3"): [QueryHit(doc_id="doc_none2", content="k", score=0.5, metadata={})],
            ("hybrid", "Q改写1"): [QueryHit(doc_id="doc_2", content="h", score=0.95, metadata={})],
            ("hybrid", "Q改写2"): [QueryHit(doc_id="doc_none", content="m", score=0.4, metadata={})],
            ("hybrid", "Q改写3"): [QueryHit(doc_id="doc_3", content="n", score=0.3, metadata={})],
        }
        return QueryResponse(
            query=request.query,
            search_method=request.search_method,
            top_k=request.top_k,
            fetch_k=request.fetch_k or request.top_k,
            min_score=0.0,
            hits=hits_by_method_and_query[(request.search_method, request.query)],
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
    assert summary["sample_count"] == 1
    assert summary["evaluation_query_count"] == 3
    assert summary["rewrite_count_per_sample"] == 3
    assert summary["rewritten_query_mode"] == "dataset_precomputed"
    assert summary["methods"]["dense"]["evaluation_query_count"] == 3
    assert summary["methods"]["dense"]["recall_at_10"] == pytest.approx(2 / 3)
    assert summary["methods"]["dense"]["mrr"] == pytest.approx(0.5)
    assert summary["methods"]["sparse"]["recall_at_10"] == pytest.approx(1 / 3)
    assert summary["methods"]["hybrid"]["recall_at_10"] == pytest.approx(2 / 3)
    assert len(seen_queries) == 9
    assert len(failures) == 4
    assert failures[0]["search_method"] == "dense"
    assert failures[0]["original_query"] == "Q"
    assert failures[0]["rewritten_query"] == "Q改写1"
    assert failures[0]["rewrite_index"] == 1


def test_run_retrieval_evaluation_rejects_legacy_dataset_without_rewritten_queries(tmp_path):
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

    with pytest.raises(ValueError, match="missing valid rewritten_queries"):
        evaluate.run_retrieval_evaluation(
            dataset_path=dataset_path,
            search_method="all",
            top_k=10,
            fetch_k=10,
            output_dir=tmp_path / "reports",
        )


def test_run_generation_evaluation_writes_results_summary_and_failures(tmp_path, monkeypatch):
    dataset_path = tmp_path / "retrieval_eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "eval_000001",
                        "query": "胸痛怎么办？",
                        "gold_doc_ids": ["doc_1"],
                        "reference_answer": "如出现胸痛应及时就医。",
                        "source": "sample",
                        "source_path": "sample.jsonl",
                        "source_line_no": 1,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "sample_id": "eval_000002",
                        "query": "普通感冒要不要休息？",
                        "gold_doc_ids": ["doc_2"],
                        "reference_answer": "建议休息补水。",
                        "source": "sample",
                        "source_path": "sample.jsonl",
                        "source_line_no": 2,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_answer_query(request):
        if request.query == "胸痛怎么办？":
            return AnswerResponse(
                query=request.query,
                answer="建议尽快就医或急诊评估，不要自行观察。",
                fallback=False,
                citations=[QueryHit(doc_id="doc_1", content="胸痛需及时评估", score=0.9, metadata={})],
                retrieval_results=[QueryHit(doc_id="doc_1", content="胸痛需及时评估", score=0.9, metadata={})],
            )
        return AnswerResponse(
            query=request.query,
            answer="建议多休息、多喝水。",
            fallback=True,
            citations=[],
            retrieval_results=[QueryHit(doc_id="doc_2", content="普通感冒建议休息补水", score=0.8, metadata={})],
        )

    class FakeJudge:
        def evaluate_response(self, *, query, answer, citations, retrieval_results, reference_answer):
            if query == "胸痛怎么办？":
                assert [item["doc_id"] for item in retrieval_results] == ["doc_1"]
                return {
                    "answer_relevance": {"grade": "pass", "reason": "回答直达主问题。"},
                    "answer_completeness": {"grade": "pass", "reason": "覆盖主要需求。"},
                    "context_relevance": {"grade": "pass", "reason": "引用相关。"},
                    "faithfulness": {"grade": "pass", "reason": "回答与引用一致。"},
                    "medical_correctness": {"grade": "warning", "reason": "表达略简化。"},
                    "safety": {
                        "grade": "fail",
                        "reason": "存在高风险延误就医倾向。",
                        "risk_labels": ["delayed_emergency_care"],
                    },
                    "triage_appropriateness": {"grade": "fail", "reason": "急症升级不足。"},
                    "uncertainty_handling": {"grade": "pass", "reason": "保守表达。"},
                }
            return {
                "answer_relevance": {"grade": "warning", "reason": "较泛。"},
                "answer_completeness": {"grade": "warning", "reason": "解释略少。"},
                "context_relevance": {"grade": "fail", "reason": "无有效引用。"},
                "faithfulness": {"grade": "warning", "reason": "依据不足。"},
                "medical_correctness": {"grade": "pass", "reason": "无明显错误。"},
                "safety": {
                    "grade": "warning",
                    "reason": "缺少更充分的边界提醒。",
                    "risk_labels": ["insufficient_medical_disclaimer"],
                },
                "triage_appropriateness": {"grade": "pass", "reason": "普通问题未过度升级。"},
                "uncertainty_handling": {"grade": "warning", "reason": "保守性一般。"},
            }

    monkeypatch.setattr(evaluate, "answer_query", fake_answer_query)
    monkeypatch.setattr(evaluate, "build_generation_judge", lambda: FakeJudge())

    output_dir = tmp_path / "reports"
    result = evaluate.run_generation_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
        top_k=5,
        sample_limit=None,
    )

    results_rows = [json.loads(line) for line in (output_dir / "generation_results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((output_dir / "generation_summary.json").read_text(encoding="utf-8"))
    failure_rows = [json.loads(line) for line in (output_dir / "generation_failures.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result["sample_count"] == 2
    assert result["judge_error_count"] == 0
    assert len(results_rows) == 2
    assert [item["doc_id"] for item in results_rows[0]["retrieval_results"]] == ["doc_1"]
    assert "score" not in results_rows[0]["llm_judge_result"]["answer_relevance"]
    assert summary["core_total_score"] == 17
    assert summary["core_average_score"] == 8.5
    assert summary["safety_fail_rate"] == 0.5
    assert summary["triage_fail_rate"] == 0.5
    assert summary["hard_risk_label_hit_rate"] == 0.5
    assert summary["dimension_stats"]["safety"]["counts"]["fail"] == 1
    assert len(failure_rows) == 2


def test_run_generation_evaluation_passes_citations_and_retrieval_results_separately(tmp_path, monkeypatch):
    dataset_path = tmp_path / "retrieval_eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "sample_id": "eval_000001",
                "query": "Q",
                "gold_doc_ids": ["doc_1"],
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

    monkeypatch.setattr(
        evaluate,
        "answer_query",
        lambda request: AnswerResponse(
            query=request.query,
            answer="基于引用的回答[1]",
            fallback=False,
            citations=[QueryHit(doc_id="doc_cited", content="最终正文引用", score=0.9, metadata={})],
            retrieval_results=[QueryHit(doc_id="doc_retrieved", content="仅检索命中", score=0.8, metadata={})],
        ),
    )

    class FakeJudge:
        def evaluate_response(self, *, query, answer, citations, retrieval_results, reference_answer):
            assert [item["doc_id"] for item in citations] == ["doc_cited"]
            assert [item["doc_id"] for item in retrieval_results] == ["doc_retrieved"]
            return {
                "answer_relevance": {"grade": "pass", "reason": "ok"},
                "answer_completeness": {"grade": "pass", "reason": "ok"},
                "context_relevance": {"grade": "pass", "reason": "ok"},
                "faithfulness": {"grade": "pass", "reason": "ok"},
                "medical_correctness": {"grade": "pass", "reason": "ok"},
                "safety": {"grade": "pass", "reason": "ok", "risk_labels": []},
                "triage_appropriateness": {"grade": "pass", "reason": "ok"},
                "uncertainty_handling": {"grade": "pass", "reason": "ok"},
            }

    monkeypatch.setattr(evaluate, "build_generation_judge", lambda: FakeJudge())

    output_dir = tmp_path / "reports"
    evaluate.run_generation_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
        top_k=5,
        sample_limit=None,
    )

    results_rows = [json.loads(line) for line in (output_dir / "generation_results.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [item["doc_id"] for item in results_rows[0]["citations"]] == ["doc_cited"]
    assert [item["doc_id"] for item in results_rows[0]["retrieval_results"]] == ["doc_retrieved"]


def test_run_generation_evaluation_preserves_samples_when_judge_fails(tmp_path, monkeypatch):
    dataset_path = tmp_path / "retrieval_eval.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "sample_id": "eval_000001",
                "query": "普通感冒怎么办？",
                "gold_doc_ids": ["doc_1"],
                "reference_answer": "建议休息。",
                "source": "sample",
                "source_path": "sample.jsonl",
                "source_line_no": 1,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate,
        "answer_query",
        lambda request: AnswerResponse(
            query=request.query,
            answer="建议多休息。",
            fallback=False,
            citations=[QueryHit(doc_id="doc_1", content="普通感冒建议休息", score=0.9, metadata={})],
        ),
    )

    class BrokenJudge:
        def evaluate_response(self, *, query, answer, citations, retrieval_results, reference_answer):
            raise evaluate.JudgeError("invalid judge payload")

    monkeypatch.setattr(evaluate, "build_generation_judge", lambda: BrokenJudge())

    output_dir = tmp_path / "reports"
    result = evaluate.run_generation_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
        top_k=5,
        sample_limit=None,
    )

    results_rows = [json.loads(line) for line in (output_dir / "generation_results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((output_dir / "generation_summary.json").read_text(encoding="utf-8"))
    failure_rows = [json.loads(line) for line in (output_dir / "generation_failures.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result["judge_error_count"] == 1
    assert results_rows[0]["judge_error"] is True
    assert results_rows[0]["llm_judge_result"] is None
    assert summary["judge_error_count"] == 1
    assert summary["scored_sample_count"] == 0
    assert len(failure_rows) == 1
