from pathlib import Path

import pytest

from app.ingest import InfrastructureCircuitBreakerOpen, MongoUnavailable
from app.ingest import service as service_module
from app.ingest.service import IngestService
from app.retrieval import VectorIndexingResult


class FakeRepository:
    def __init__(self):
        self.run = None
        self.batches = {}
        self.events = []

    def start_run(self, source_path, collection_name, parameters, force_reingest):
        if self.run is None or force_reingest:
            self.run = {
                "ingest_run_id": "run_1",
                "source_path": source_path,
                "collection_name": collection_name,
                "parameters": parameters,
                "lease_owner": "lease_1",
                "status": "queued",
                "stage": "queued",
                "cancel_requested": False,
                "total_lines": 0,
                "processed_batches": 0,
                "skipped_succeeded_batches": 0,
                "embedding_seconds_total": 0.0,
                "embedding_seconds_last_batch": 0.0,
                "last_scanned_line": 0,
            }
        else:
            self.run["lease_owner"] = "lease_1"
            self.run["status"] = "queued"
            self.run["stage"] = "queued"
            self.run["cancel_requested"] = False
        return self.run

    def get_or_create_batch(self, ingest_run_id, start_line, end_line):
        key = (start_line, end_line)
        self.batches.setdefault(
            key,
            {
                "batch_id": f"batch_{start_line}_{end_line}",
                "start_line": start_line,
                "end_line": end_line,
                "status": "pending",
                "attempt_count": 0,
            },
        )
        return self.batches[key]

    def renew_lease(self, ingest_run_id, lease_owner):
        return None

    def mark_run_running(self, ingest_run_id, lease_owner, stage="counting_lines"):
        self.run.update(
            status="running",
            stage=stage,
            lease_owner=lease_owner,
            total_lines=0,
            last_scanned_line=0,
            processed_batches=0,
            skipped_succeeded_batches=0,
            embedding_seconds_total=0.0,
            embedding_seconds_last_batch=0.0,
        )
        return self.run

    def update_run_stage(self, ingest_run_id, lease_owner, stage, total_lines=None):
        self.run.update(stage=stage, lease_owner=lease_owner)
        if total_lines is not None:
            self.run["total_lines"] = total_lines
        return self.run

    def is_cancel_requested(self, ingest_run_id):
        return bool(self.run.get("cancel_requested", False))

    def mark_run_cancelled(self, ingest_run_id, lease_owner, reason):
        self.run.update(status="cancelled", stage="finished", lease_owner=None, error=reason)
        return self.run

    def update_run_progress(
        self,
        ingest_run_id,
        lease_owner,
        last_scanned_line,
        processed_batches_inc,
        skipped_succeeded_batches_inc,
        embedding_seconds_total_inc,
        embedding_seconds_last_batch,
    ):
        self.run["lease_owner"] = lease_owner
        self.run["last_scanned_line"] = max(self.run.get("last_scanned_line", 0), last_scanned_line)
        self.run["processed_batches"] = self.run.get("processed_batches", 0) + processed_batches_inc
        self.run["skipped_succeeded_batches"] = (
            self.run.get("skipped_succeeded_batches", 0) + skipped_succeeded_batches_inc
        )
        self.run["embedding_seconds_total"] = (
            self.run.get("embedding_seconds_total", 0.0) + embedding_seconds_total_inc
        )
        self.run["embedding_seconds_last_batch"] = embedding_seconds_last_batch
        return self.run

    def mark_batch_running(self, batch_id):
        batch = self._find(batch_id)
        batch["status"] = "running"
        batch["attempt_count"] += 1

    def mark_batch_succeeded(self, batch_id, raw_count, cleaned_count, document_count, indexed_count, existing_count):
        self._find(batch_id).update(
            status="succeeded",
            raw_count=raw_count,
            cleaned_count=cleaned_count,
            document_count=document_count,
            indexed_count=indexed_count,
            existing_count=existing_count,
            error=None,
        )

    def mark_batch_failed(self, batch_id, error, **counts):
        self._find(batch_id).update(status="failed", error=error, **counts)

    def add_event(self, ingest_run_id, event_type, payload):
        self.events.append((event_type, payload))

    def finalize_run(self, ingest_run_id, lease_owner):
        ordered = [self.batches[key] for key in sorted(self.batches)]
        checkpoint = 0
        for batch in ordered:
            if batch["status"] != "succeeded":
                break
            checkpoint = batch["end_line"]
        failed = [batch for batch in ordered if batch["status"] == "failed"]
        self.run.update(
            status="completed_with_errors" if failed else "completed",
            stage="finished",
            execution_outcome=self._execution_outcome(),
            raw_count=sum(batch.get("raw_count", 0) for batch in ordered),
            cleaned_count=sum(batch.get("cleaned_count", 0) for batch in ordered),
            document_count=sum(batch.get("document_count", 0) for batch in ordered),
            succeeded_batches=sum(batch["status"] == "succeeded" for batch in ordered),
            pending_batches=0,
            failed_batches=len(failed),
            checkpoint_line=checkpoint,
            failed_batch_details=failed,
        )
        return self.run

    def fail_run(self, ingest_run_id, lease_owner, error):
        self.run.update(status="failed", stage="finished", error=error, lease_owner=None)

    def get_status(self, ingest_run_id):
        failed = [
            {
                "batch_id": batch["batch_id"],
                "start_line": batch["start_line"],
                "end_line": batch["end_line"],
                "attempt_count": batch.get("attempt_count", 0),
                "error": batch.get("error"),
            }
            for batch in self.batches.values()
            if batch["status"] == "failed"
        ]
        run = dict(self.run)
        run.setdefault("failed_batch_details", failed)
        run.setdefault("lines_per_second", None)
        run.setdefault("batches_per_second", None)
        run.setdefault("eta_seconds", None)
        run.setdefault("current_scanned_line", run.get("last_scanned_line", 0))
        run.setdefault("execution_outcome", self._execution_outcome())
        return run

    def _find(self, batch_id):
        return next(batch for batch in self.batches.values() if batch["batch_id"] == batch_id)

    def _execution_outcome(self):
        processed_batches = int(self.run.get("processed_batches", 0) or 0)
        skipped_succeeded_batches = int(self.run.get("skipped_succeeded_batches", 0) or 0)
        if processed_batches <= 0:
            return "not_started"
        if processed_batches == skipped_succeeded_batches:
            return "skipped_all"
        return "processed"


class FailFirstBatchOnceIndexer:
    def __init__(self):
        self.failed_once = False
        self.flushed = 0
        self.verified_doc_ids = []

    def index_in_batches(self, documents):
        if documents[0].question == "Q1" and not self.failed_once:
            self.failed_once = True
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=0,
                failed_count=len(documents),
                failed_batches=1,
                milvus_failed_batches=1,
                failure_stage="milvus",
                errors=["temporary Milvus failure"],
            )
        return VectorIndexingResult(batch_size=len(documents), indexed_count=len(documents))

    def flush(self):
        self.flushed += 1

    def verify_persisted_doc_ids(self, doc_ids):
        self.verified_doc_ids.append(tuple(doc_ids))


class UnavailableRepository:
    def start_run(self, source_path, collection_name, parameters, force_reingest):
        raise MongoUnavailable("MongoDB unavailable")


class SequenceIndexer:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.flushed = 0
        self.verified_doc_ids = []

    def index_in_batches(self, documents):
        outcome = next(self.outcomes, "succeeded")
        if outcome == "milvus_failed":
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=0,
                failed_count=len(documents),
                failed_batches=1,
                milvus_failed_batches=1,
                failure_stage="milvus",
                errors=["Milvus unavailable"],
            )
        if outcome == "failed":
            return VectorIndexingResult(
                batch_size=len(documents),
                indexed_count=0,
                failed_count=len(documents),
                failed_batches=1,
                failure_stage="embedding",
                errors=["embedding failed"],
            )
        return VectorIndexingResult(batch_size=len(documents), indexed_count=len(documents))

    def flush(self):
        self.flushed += 1

    def verify_persisted_doc_ids(self, doc_ids):
        self.verified_doc_ids.append(tuple(doc_ids))


class FlushFailingIndexer(SequenceIndexer):
    def flush(self):
        self.flushed += 1
        raise RuntimeError("flush timeout")


def write_source(tmp_path: Path, count: int) -> Path:
    source = tmp_path / "sample.jsonl"
    source.write_text(
        "".join(f'{{"question":"Q{index}","answer":"A{index}"}}\n' for index in range(1, count + 1)),
        encoding="utf-8",
    )
    return source


def test_ingest_retries_failed_gap_without_reprocessing_successful_batches(tmp_path: Path):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        '{"question":"Q1","answer":"A1"}\n{"question":"Q2","answer":"A2"}\n',
        encoding="utf-8",
    )
    repository = FakeRepository()
    indexer = FailFirstBatchOnceIndexer()
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    first = service.ingest(str(source))

    assert first["status"] == "completed_with_errors"
    assert first["failed_batches"] == 1
    assert first["checkpoint_line"] == 0
    assert repository.batches[(1, 1)]["attempt_count"] == 1
    assert repository.batches[(2, 2)]["attempt_count"] == 1

    second = service.ingest(str(source))

    assert second["status"] == "completed"
    assert second["failed_batches"] == 0
    assert second["checkpoint_line"] == 2
    assert repository.batches[(1, 1)]["attempt_count"] == 2
    assert repository.batches[(2, 2)]["attempt_count"] == 1
    assert sum(event[0] == "batch_succeeded" for event in repository.events) == 2
    assert second["processed_batches"] == 2
    assert second["skipped_succeeded_batches"] == 1
    assert second["execution_outcome"] == "processed"


def test_ingest_reports_skipped_all_when_every_batch_was_already_succeeded(tmp_path: Path):
    source = write_source(tmp_path, 2)
    repository = FakeRepository()
    service = IngestService(
        repository=repository,
        indexer_factory=lambda: SequenceIndexer(["succeeded", "succeeded"]),
        batch_size=1,
    )

    first = service.ingest(str(source))
    assert first["execution_outcome"] == "processed"

    second = service.ingest(str(source))

    assert second["status"] == "completed"
    assert second["processed_batches"] == 2
    assert second["skipped_succeeded_batches"] == 2
    assert second["execution_outcome"] == "skipped_all"


def test_mongodb_failure_prevents_indexer_creation(tmp_path: Path):
    source = tmp_path / "sample.jsonl"
    source.write_text('{"question":"Q1","answer":"A1"}\n', encoding="utf-8")
    indexer_created = False

    def indexer_factory():
        nonlocal indexer_created
        indexer_created = True
        raise AssertionError("Indexer must not be created when MongoDB is unavailable")

    service = IngestService(repository=UnavailableRepository(), indexer_factory=indexer_factory, batch_size=1)

    with pytest.raises(MongoUnavailable):
        service.ingest(source_path=str(source))

    assert indexer_created is False


def test_three_consecutive_milvus_failures_open_circuit_breaker(tmp_path: Path):
    source = write_source(tmp_path, 4)
    repository = FakeRepository()
    indexer = SequenceIndexer(["milvus_failed", "milvus_failed", "milvus_failed", "succeeded"])
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    with pytest.raises(InfrastructureCircuitBreakerOpen):
        service.ingest(str(source))

    assert sorted(repository.batches) == [(1, 1), (2, 2), (3, 3)]
    assert all(batch["status"] == "failed" for batch in repository.batches.values())
    assert repository.run["status"] == "failed"
    assert repository.run["lease_owner"] is None
    event_type, payload = next(event for event in repository.events if event[0] == "circuit_breaker_opened")
    assert event_type == "circuit_breaker_opened"
    assert payload["failure_threshold"] == 3
    assert payload["failure_stage"] == "milvus"
    assert payload["start_line"] == 3
    assert payload["end_line"] == 3


def test_successful_batch_resets_infrastructure_failure_counter(tmp_path: Path):
    source = write_source(tmp_path, 5)
    repository = FakeRepository()
    indexer = SequenceIndexer(["milvus_failed", "milvus_failed", "succeeded", "milvus_failed", "milvus_failed"])
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    result = service.ingest(str(source))

    assert result["status"] == "completed_with_errors"
    assert len(repository.batches) == 5
    assert not any(event[0] == "circuit_breaker_opened" for event in repository.events)


def test_non_infrastructure_failure_resets_infrastructure_failure_counter(tmp_path: Path, monkeypatch):
    source = write_source(tmp_path, 5)
    repository = FakeRepository()
    indexer = SequenceIndexer(["milvus_failed", "milvus_failed", "milvus_failed", "succeeded"])
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    original_build_documents = service_module.build_documents

    def flaky_build_documents(cleaned_records, batch_id=None):
        if cleaned_records[0]["question"] == "Q2":
            raise ValueError("invalid cleaned record")
        return original_build_documents(cleaned_records, batch_id=batch_id)

    monkeypatch.setattr(service_module, "build_documents", flaky_build_documents)

    result = service.ingest(str(source))

    assert result["status"] == "completed_with_errors"
    assert len(repository.batches) == 5
    assert not any(event[0] == "circuit_breaker_opened" for event in repository.events)


def test_ingest_flushes_once_after_all_batches_finish(tmp_path: Path):
    source = write_source(tmp_path, 3)
    repository = FakeRepository()
    indexer = SequenceIndexer(["succeeded", "succeeded", "succeeded"])
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    result = service.ingest(str(source))

    assert result["status"] == "completed"
    assert indexer.flushed == 1
    assert len(indexer.verified_doc_ids) == 1
    assert len(indexer.verified_doc_ids[0]) == 3


def test_failed_final_flush_marks_written_batches_failed(tmp_path: Path):
    source = write_source(tmp_path, 2)
    repository = FakeRepository()
    indexer = FlushFailingIndexer(["succeeded", "succeeded"])
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    with pytest.raises(RuntimeError, match="Deferred Milvus flush failed"):
        service.ingest(str(source))

    assert repository.run["status"] == "failed"
    assert all(batch["status"] == "failed" for batch in repository.batches.values())
    assert any(
        event_type == "batch_failed" and payload.get("deferred_flush_failure") is True
        for event_type, payload in repository.events
    )


def test_three_consecutive_embedding_failures_open_circuit_breaker(tmp_path: Path):
    source = write_source(tmp_path, 4)
    repository = FakeRepository()
    indexer = SequenceIndexer(["failed", "failed", "failed", "succeeded"])
    service = IngestService(repository=repository, indexer_factory=lambda: indexer, batch_size=1)

    with pytest.raises(InfrastructureCircuitBreakerOpen):
        service.ingest(str(source))

    assert sorted(repository.batches) == [(1, 1), (2, 2), (3, 3)]
    event_type, payload = next(event for event in repository.events if event[0] == "circuit_breaker_opened")
    assert event_type == "circuit_breaker_opened"
    assert payload["failure_stage"] == "embedding"


def test_ingest_can_resume_after_infrastructure_circuit_breaker(tmp_path: Path):
    source = write_source(tmp_path, 4)
    repository = FakeRepository()
    failing_service = IngestService(
        repository=repository,
        indexer_factory=lambda: SequenceIndexer(["milvus_failed", "milvus_failed", "milvus_failed"]),
        batch_size=1,
    )

    with pytest.raises(InfrastructureCircuitBreakerOpen):
        failing_service.ingest(str(source))

    recovered_service = IngestService(
        repository=repository,
        indexer_factory=lambda: SequenceIndexer(["succeeded"] * 4),
        batch_size=1,
    )
    result = recovered_service.ingest(str(source))

    assert result["status"] == "completed"
    assert result["checkpoint_line"] == 4
    assert all(batch["status"] == "succeeded" for batch in repository.batches.values())


def test_ingest_marks_run_cancelled_when_cancel_requested_before_execution(tmp_path: Path):
    source = write_source(tmp_path, 2)

    class CancelledRepository(FakeRepository):
        def is_cancel_requested(self, ingest_run_id):
            return True

    repository = CancelledRepository()
    service = IngestService(repository=repository, indexer_factory=lambda: SequenceIndexer(["succeeded"]), batch_size=1)

    result = service.ingest(str(source))

    assert result["status"] == "cancelled"
    assert repository.batches == {}
