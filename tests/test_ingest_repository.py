from datetime import datetime, timedelta

import pytest

from app.ingest import IngestLeaseConflict
from app.ingest import repository as repository_module
from app.ingest.repository import MongoIngestRepository


class FakeRuns:
    def __init__(self):
        self.documents = []

    def find_one(self, query, sort=None):
        matches = [
            document
            for document in self.documents
            if document["source_path"] == query["source_path"]
            and document.get("collection_name") == query.get("collection_name")
        ]
        if sort:
            matches.sort(key=lambda document: document["created_at"], reverse=True)
        return matches[0] if matches else None

    def insert_one(self, document):
        self.documents.append(document)

    def find_one_and_update(self, query, update, return_document):
        for document in self.documents:
            if document["ingest_run_id"] != query["ingest_run_id"]:
                continue
            if document.get("lease_expires_at") and document["lease_expires_at"] > FIXED_NOW:
                continue
            document.update(update["$set"])
            return document
        return None


class FakeBatches:
    def __init__(self):
        self.documents = []

    def update_many(self, query, update):
        for document in self.documents:
            if document["ingest_run_id"] == query["ingest_run_id"] and document["status"] == query["status"]:
                document.update(update["$set"])


FIXED_NOW = datetime(2026, 6, 2, 12, 0, 0)


def make_repository(monkeypatch):
    repository = object.__new__(MongoIngestRepository)
    repository.runs = FakeRuns()
    repository.batches = FakeBatches()
    repository.lease_seconds = 300
    repository.ping = lambda: True
    repository.add_event = lambda *args, **kwargs: None
    monkeypatch.setattr(repository_module, "_utcnow", lambda: FIXED_NOW)
    return repository


def test_active_source_lease_conflicts_and_expired_lease_resumes(monkeypatch):
    repository = make_repository(monkeypatch)
    first = repository.start_run("source.jsonl", "collection_a", {"flush_to_milvus": True}, force_reingest=False)
    repository.batches.documents.append(
        {"ingest_run_id": first["ingest_run_id"], "status": "running"}
    )

    with pytest.raises(IngestLeaseConflict):
        repository.start_run("source.jsonl", "collection_a", {"flush_to_milvus": True}, force_reingest=False)

    first["lease_expires_at"] = FIXED_NOW - timedelta(seconds=1)
    resumed = repository.start_run("source.jsonl", "collection_a", {"flush_to_milvus": True}, force_reingest=False)

    assert resumed["ingest_run_id"] == first["ingest_run_id"]
    assert repository.batches.documents[0]["status"] == "pending"


def test_force_reingest_still_rejects_active_source_lease(monkeypatch):
    repository = make_repository(monkeypatch)
    repository.start_run("source.jsonl", "collection_a", {"flush_to_milvus": True}, force_reingest=False)

    with pytest.raises(IngestLeaseConflict):
        repository.start_run("source.jsonl", "collection_a", {"flush_to_milvus": True}, force_reingest=True)


def test_same_source_path_uses_independent_runs_per_collection(monkeypatch):
    repository = make_repository(monkeypatch)

    first = repository.start_run("source.jsonl", "collection_a", {"flush_to_milvus": True}, force_reingest=False)
    second = repository.start_run("source.jsonl", "collection_b", {"flush_to_milvus": True}, force_reingest=False)

    assert first["ingest_run_id"] != second["ingest_run_id"]
    assert first["collection_name"] == "collection_a"
    assert second["collection_name"] == "collection_b"


def test_find_latest_run_for_source_path_prefers_active_run(monkeypatch):
    class LookupRuns:
        def find_one(self, query, projection=None, sort=None):
            if query.get("status"):
                return {"ingest_run_id": "run_active"}
            return {"ingest_run_id": "run_latest"}

    repository = object.__new__(MongoIngestRepository)
    repository.runs = LookupRuns()
    repository.ping = lambda: True
    repository.get_status = lambda ingest_run_id: {"ingest_run_id": ingest_run_id, "status": "running"}

    state = repository.find_latest_run_for_source_path("source.jsonl", "collection_a")

    assert state["ingest_run_id"] == "run_active"


def test_get_status_marks_expired_running_run_as_interrupted(monkeypatch):
    class EmptySorted:
        def sort(self, *args, **kwargs):
            return []

    class StatusRuns:
        def find_one(self, query, projection=None):
            return {
                "ingest_run_id": "run_stale",
                "status": "running",
                "stage": "ingesting",
                "lease_owner": "lease_1",
                "lease_expires_at": FIXED_NOW - timedelta(seconds=1),
                "source_path": "source.jsonl",
                "collection_name": "collection_a",
                "processed_batches": 1,
                "skipped_succeeded_batches": 0,
                "last_scanned_line": 32,
                "total_lines": 100,
            }

    class StatusBatches:
        def find(self, query, projection=None):
            return EmptySorted()

    repository = object.__new__(MongoIngestRepository)
    repository.runs = StatusRuns()
    repository.batches = StatusBatches()
    repository.ping = lambda: True
    monkeypatch.setattr(repository_module, "_utcnow", lambda: FIXED_NOW)

    state = repository.get_status("run_stale")

    assert state["status"] == "interrupted"
    assert state["abandoned"] is True
    assert state["stale"] is True
    assert state["lease_owner"] is None


def test_request_cancel_finishes_interrupted_run_without_owner(monkeypatch):
    class CancelRuns:
        def __init__(self):
            self.document = {
                "ingest_run_id": "run_interrupted",
                "status": "interrupted",
                "stage": "interrupted",
                "lease_owner": None,
                "source_path": "source.jsonl",
                "collection_name": "collection_a",
            }

        def find_one(self, query):
            if query["ingest_run_id"] == self.document["ingest_run_id"]:
                return dict(self.document)
            return None

        def find_one_and_update(self, query, update, return_document=None):
            if query.get("lease_owner") is None and self.document.get("lease_owner") is not None:
                return None
            self.document.update(update.get("$set", {}))
            for key in update.get("$unset", {}):
                self.document.pop(key, None)
            return dict(self.document)

    repository = object.__new__(MongoIngestRepository)
    repository.runs = CancelRuns()
    repository.ping = lambda: True
    repository._enrich_run_status = lambda run: run
    events = []
    repository.add_event = lambda ingest_run_id, event_type, payload: events.append((event_type, payload))
    monkeypatch.setattr(repository_module, "_utcnow", lambda: FIXED_NOW)

    state = repository.request_cancel("run_interrupted")

    assert state["status"] == "cancelled"
    assert events[-1][0] == "run_cancelled"


def test_reconcile_interrupted_runs_marks_orphaned_active_runs(monkeypatch):
    class ReconcileRuns:
        def __init__(self):
            self.documents = [
                {
                    "ingest_run_id": "run_1",
                    "status": "running",
                    "stage": "ingesting",
                    "lease_owner": "lease_1",
                },
                {
                    "ingest_run_id": "run_2",
                    "status": "queued",
                    "stage": "queued",
                    "lease_owner": "lease_2",
                },
                {
                    "ingest_run_id": "run_3",
                    "status": "completed",
                    "stage": "finished",
                    "lease_owner": None,
                },
            ]

        def find(self, query, projection=None):
            statuses = set(query["status"]["$in"])
            return [
                dict(document)
                for document in self.documents
                if document["status"] in statuses and document.get("lease_owner") is not None
            ]

        def update_one(self, query, update):
            for document in self.documents:
                if document["ingest_run_id"] != query["ingest_run_id"]:
                    continue
                document.update(update.get("$set", {}))
                for key in update.get("$unset", {}):
                    document.pop(key, None)
                return

    repository = object.__new__(MongoIngestRepository)
    repository.runs = ReconcileRuns()
    repository.ping = lambda: True
    events = []
    repository.add_event = lambda ingest_run_id, event_type, payload: events.append((ingest_run_id, event_type, payload))
    monkeypatch.setattr(repository_module, "_utcnow", lambda: FIXED_NOW)

    reconciled = repository.reconcile_interrupted_runs()

    assert reconciled == 2
    assert repository.runs.documents[0]["status"] == "interrupted"
    assert repository.runs.documents[1]["status"] == "interrupted"
    assert all(event[1] == "run_interrupted" for event in events)
