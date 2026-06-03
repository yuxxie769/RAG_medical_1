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
