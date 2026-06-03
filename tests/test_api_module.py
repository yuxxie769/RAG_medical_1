from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import app, IngestRequest, QueryHit, QueryRequest, QueryResponse
from app.core.models import RetrievalResult
from app.ingest import IngestLeaseConflict, MongoUnavailable
from app.retrieval import MilvusOperationError


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
    assert not hasattr(request, "resume_from_line")


def test_ingest_returns_202_and_background_urls(monkeypatch):
    class AsyncReadyService:
        def __init__(self):
            self.prepared = None
            self.executed = None

        def prepare_ingest(self, **kwargs):
            self.prepared = kwargs
            return SimpleNamespace(
                ingest_run_id="run_1",
                source_path="C:\\ingest\\sample.jsonl",
            )

        def execute_prepared_ingest(self, prepared_run):
            self.executed = prepared_run

    service = AsyncReadyService()
    monkeypatch.setitem(api_main._DATASTORE, "ingest_service", service)

    response = TestClient(app).post("/ingest", json={"source_path": "sample.jsonl"})

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["ingest_run_id"] == "run_1"
    assert payload["status_url"] == "/ingest/status?ingest_run_id=run_1"
    assert payload["events_url"] == "/ingest/run_1/events"
    assert payload["cancel_url"] == "/ingest/run_1"
    assert service.prepared["source_path"] == "sample.jsonl"
    assert service.executed.ingest_run_id == "run_1"


def test_ingest_returns_503_when_mongodb_is_unavailable(monkeypatch):
    class UnavailableService:
        def prepare_ingest(self, **kwargs):
            raise MongoUnavailable("MongoDB unavailable")

    monkeypatch.setitem(api_main._DATASTORE, "ingest_service", UnavailableService())

    response = TestClient(app).post("/ingest", json={"source_path": "sample.jsonl"})

    assert response.status_code == 503


def test_ingest_returns_409_when_source_lease_is_active(monkeypatch):
    class ConflictingService:
        def prepare_ingest(self, **kwargs):
            raise IngestLeaseConflict("lease active")

    monkeypatch.setitem(api_main._DATASTORE, "ingest_service", ConflictingService())

    response = TestClient(app).post("/ingest", json={"source_path": "sample.jsonl"})

    assert response.status_code == 409


def test_cancel_ingest_returns_404_when_run_is_missing(monkeypatch):
    class MissingRunRepository:
        def request_cancel(self, ingest_run_id):
            return None

    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", MissingRunRepository())

    response = TestClient(app).delete("/ingest/run_missing")

    assert response.status_code == 404


def test_cancel_ingest_returns_updated_status(monkeypatch):
    class CancelRepository:
        def request_cancel(self, ingest_run_id):
            return {
                "ingest_run_id": ingest_run_id,
                "status": "cancelling",
                "stage": "cancelling",
                "execution_outcome": "not_started",
                "source_path": "sample.jsonl",
                "cancel_requested": True,
                "failed_batch_details": [],
            }

    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", CancelRepository())

    response = TestClient(app).delete("/ingest/run_1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "cancelling"
    assert payload["cancel_requested"] is True
    assert payload["execution_outcome"] == "not_started"


def test_ingest_status_includes_execution_outcome(monkeypatch):
    class StatusRepository:
        def get_status(self, ingest_run_id):
            return {
                "ingest_run_id": ingest_run_id,
                "status": "completed",
                "stage": "finished",
                "execution_outcome": "skipped_all",
                "source_path": "sample.jsonl",
                "processed_batches": 2,
                "skipped_succeeded_batches": 2,
                "failed_batch_details": [],
            }

    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", StatusRepository())

    response = TestClient(app).get("/ingest/status", params={"ingest_run_id": "run_1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["execution_outcome"] == "skipped_all"


def test_ingest_events_stream_progress_until_terminal_state(monkeypatch):
    class EventsRepository:
        def __init__(self):
            self.calls = 0

        def get_status(self, ingest_run_id):
            self.calls += 1
            if self.calls == 1:
                return {
                    "ingest_run_id": ingest_run_id,
                    "status": "running",
                    "stage": "ingesting",
                    "source_path": "sample.jsonl",
                    "updated_at": "2026-06-02T12:00:00Z",
                    "failed_batch_details": [],
                }
            return {
                "ingest_run_id": ingest_run_id,
                "status": "completed",
                "stage": "finished",
                "source_path": "sample.jsonl",
                "updated_at": "2026-06-02T12:00:01Z",
                "failed_batch_details": [],
            }

    monkeypatch.setitem(api_main._DATASTORE, "ingest_repository", EventsRepository())

    with TestClient(app).stream("GET", "/ingest/run_1/events") as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: progress" in body
    assert '"status": "completed"' in body


def test_query_returns_503_when_milvus_is_unavailable(monkeypatch):
    class BrokenStore:
        def count(self):
            raise MilvusOperationError("Milvus get_collection_stats failed or timed out after 3.0s: timeout")

    monkeypatch.setitem(api_main._DATASTORE, "store", BrokenStore())

    response = TestClient(app).post("/query", json={"query": "A"})

    assert response.status_code == 503


def test_query_returns_rerank_metadata(monkeypatch):
    class ReadyStore:
        def count(self):
            return 1

    class FakeRetriever:
        def retrieve(self, **kwargs):
            return [
                RetrievalResult(
                    doc_id="doc_1",
                    content="Q: A\nA: B",
                    score=0.91,
                    metadata={
                        "retrieval_score": 0.41,
                        "rerank_score": 0.91,
                        "rerank_applied": True,
                        "rerank_provider": "openai_compatible",
                    },
                )
            ]

    monkeypatch.setitem(api_main._DATASTORE, "store", ReadyStore())
    monkeypatch.setitem(api_main._DATASTORE, "retriever", FakeRetriever())

    response = TestClient(app).post("/query", json={"query": "A", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"][0]["score"] == 0.91
    assert payload["hits"][0]["metadata"]["retrieval_score"] == 0.41
    assert payload["hits"][0]["metadata"]["rerank_applied"] is True


def test_answer_inherits_rerank_metadata_from_query_results(monkeypatch):
    class ReadyStore:
        def count(self):
            return 1

    class FakeRetriever:
        def retrieve(self, **kwargs):
            return [
                RetrievalResult(
                    doc_id="doc_1",
                    content="Q: A\nA: B",
                    score=0.88,
                    metadata={
                        "retrieval_score": 0.33,
                        "rerank_score": 0.88,
                        "rerank_applied": True,
                        "rerank_provider": "openai_compatible",
                    },
                )
            ]

    class FakeGenerator:
        def generate(self, query, citations):
            return SimpleNamespace(answer="done", fallback=False)

    monkeypatch.setitem(api_main._DATASTORE, "store", ReadyStore())
    monkeypatch.setitem(api_main._DATASTORE, "retriever", FakeRetriever())
    monkeypatch.setitem(api_main._DATASTORE, "generator", FakeGenerator())

    response = TestClient(app).post("/answer", json={"query": "A", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"][0]["metadata"]["rerank_score"] == 0.88
    assert payload["citations"][0]["metadata"]["rerank_applied"] is True
