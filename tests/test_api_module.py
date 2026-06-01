from fastapi.testclient import TestClient

from app.api.main import app, IngestRequest, QueryHit, QueryRequest, QueryResponse


def test_query_request_supports_fetch_k_and_min_score():
    request = QueryRequest(query="口干怎么办？", top_k=5, fetch_k=20, min_score=0.3)

    assert request.top_k == 5
    assert request.fetch_k == 20
    assert request.min_score == 0.3
    assert request.search_method == "hybrid"


def test_query_response_includes_effective_retrieval_controls():
    hit = QueryHit(doc_id="doc_1", content="Q: A\nA: B", score=0.8, metadata={"source": "sample"})
    response = QueryResponse(
        query="A",
        search_method="hybrid",
        top_k=5,
        fetch_k=15,
        min_score=0.3,
        hits=[hit],
    )

    assert response.fetch_k == 15
    assert response.min_score == 0.3
    assert response.search_method == "hybrid"
    assert response.hits[0].score == 0.8


def test_query_request_rejects_unknown_search_method():
    response = TestClient(app).post(
        "/query",
        json={"query": "A", "search_method": "unknown"},
    )

    assert response.status_code == 422


def test_ingest_request_can_force_full_reingest_after_collection_reset():
    request = IngestRequest(source_path="tests/sample_data.jsonl", force_reingest=True)

    assert request.force_reingest is True
    assert request.resume_from_line == 1
