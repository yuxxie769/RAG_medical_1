from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import PyMongoError


TERMINAL_RUN_STATUSES = {"cancelled", "completed", "completed_with_errors", "failed"}
EXECUTION_OUTCOMES = {"not_started", "processed", "skipped_all"}


class MongoUnavailable(RuntimeError):
    """Raised when ingest state cannot be persisted safely."""


class IngestLeaseConflict(RuntimeError):
    """Raised when another request is importing the same source."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MongoIngestRepository:
    def __init__(
        self,
        uri: str,
        database: str,
        connect_timeout_ms: int = 3000,
        lease_seconds: int = 300,
        client: MongoClient | None = None,
    ):
        self.client = client or MongoClient(
            uri,
            serverSelectionTimeoutMS=connect_timeout_ms,
            connectTimeoutMS=connect_timeout_ms,
        )
        self.database = self.client[database]
        self.runs = self.database["ingest_runs"]
        self.batches = self.database["ingest_batches"]
        self.events = self.database["ingest_events"]
        self.lease_seconds = lease_seconds
        self._indexes_ready = False

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            self._ensure_indexes()
            return True
        except PyMongoError as exc:
            raise MongoUnavailable(f"MongoDB unavailable: {exc}") from exc

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        self.runs.create_index([("source_path", ASCENDING), ("collection_name", ASCENDING), ("created_at", DESCENDING)])
        self.batches.create_index(
            [("ingest_run_id", ASCENDING), ("start_line", ASCENDING), ("end_line", ASCENDING)],
            unique=True,
        )
        self.events.create_index([("ingest_run_id", ASCENDING), ("created_at", ASCENDING)])
        self._indexes_ready = True

    def start_run(
        self,
        source_path: str,
        collection_name: str | None,
        parameters: dict[str, Any],
        force_reingest: bool,
    ) -> dict[str, Any]:
        self.ping()
        now = _utcnow()
        lease_owner = uuid4().hex
        run_selector = {"source_path": source_path, "collection_name": collection_name}
        latest_run = self.runs.find_one(run_selector, sort=[("created_at", DESCENDING)])
        if latest_run and latest_run.get("lease_expires_at") and latest_run["lease_expires_at"] > now:
            raise IngestLeaseConflict(
                f"An ingest request is already running for {source_path} -> {collection_name or 'default'}"
            )

        run = None if force_reingest else latest_run
        if run is None:
            ingest_run_id = uuid4().hex
            run = {
                "ingest_run_id": ingest_run_id,
                "source_path": source_path,
                "collection_name": collection_name,
                "parameters": parameters,
                "status": "queued",
                "stage": "queued",
                "lease_owner": lease_owner,
                "lease_expires_at": now + timedelta(seconds=self.lease_seconds),
                "cancel_requested": False,
                "cancel_requested_at": None,
                "raw_count": 0,
                "cleaned_count": 0,
                "document_count": 0,
                "indexed_count": 0,
                "succeeded_batches": 0,
                "pending_batches": 0,
                "failed_batches": 0,
                "checkpoint_line": 0,
                "last_scanned_line": 0,
                "total_lines": 0,
                "processed_batches": 0,
                "skipped_succeeded_batches": 0,
                "embedding_seconds_total": 0.0,
                "embedding_seconds_last_batch": 0.0,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            self.runs.insert_one(run)
            self.add_event(
                ingest_run_id,
                "run_created",
                {"force_reingest": force_reingest, "collection_name": collection_name},
            )
            return run

        ingest_run_id = run["ingest_run_id"]
        claimed = self.runs.find_one_and_update(
            {
                "ingest_run_id": ingest_run_id,
                "$or": [
                    {"lease_expires_at": {"$lte": now}},
                    {"lease_expires_at": {"$exists": False}},
                    {"lease_owner": None},
                ],
            },
            {
                "$set": {
                    "status": "queued",
                    "stage": "queued",
                    "lease_owner": lease_owner,
                    "lease_expires_at": now + timedelta(seconds=self.lease_seconds),
                    "cancel_requested": False,
                    "cancel_requested_at": None,
                    "finished_at": None,
                    "error": None,
                    "updated_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if claimed is None:
            raise IngestLeaseConflict(
                f"An ingest request is already running for {source_path} -> {collection_name or 'default'}"
            )
        self.batches.update_many(
            {"ingest_run_id": ingest_run_id, "status": "running"},
            {"$set": {"status": "pending", "updated_at": now}},
        )
        self.add_event(ingest_run_id, "run_resumed", {"collection_name": collection_name})
        return claimed

    def mark_run_running(self, ingest_run_id: str, lease_owner: str, stage: str = "counting_lines") -> dict[str, Any]:
        now = _utcnow()
        run = self.runs.find_one_and_update(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {
                "$set": {
                    "status": "running",
                    "stage": stage,
                    "started_at": now,
                    "finished_at": None,
                    "error": None,
                    "total_lines": 0,
                    "last_scanned_line": 0,
                    "processed_batches": 0,
                    "skipped_succeeded_batches": 0,
                    "embedding_seconds_total": 0.0,
                    "embedding_seconds_last_batch": 0.0,
                    "updated_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        self.add_event(ingest_run_id, "run_started", {"stage": stage})
        return run

    def update_run_stage(
        self,
        ingest_run_id: str,
        lease_owner: str,
        stage: str,
        *,
        total_lines: int | None = None,
    ) -> dict[str, Any]:
        update: dict[str, Any] = {"stage": stage, "updated_at": _utcnow()}
        if total_lines is not None:
            update["total_lines"] = total_lines
        return self.runs.find_one_and_update(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {"$set": update},
            return_document=ReturnDocument.AFTER,
        )

    def renew_lease(self, ingest_run_id: str, lease_owner: str) -> None:
        now = _utcnow()
        result = self.runs.update_one(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {
                "$set": {
                    "lease_expires_at": now + timedelta(seconds=self.lease_seconds),
                    "updated_at": now,
                }
            },
        )
        if result.matched_count != 1:
            raise IngestLeaseConflict(f"Lost ingest lease for run {ingest_run_id}")

    def request_cancel(self, ingest_run_id: str) -> dict[str, Any] | None:
        self.ping()
        run = self.runs.find_one({"ingest_run_id": ingest_run_id})
        if run is None:
            return None
        if run.get("status") in TERMINAL_RUN_STATUSES:
            return self.get_status(ingest_run_id)

        now = _utcnow()
        updated = self.runs.find_one_and_update(
            {"ingest_run_id": ingest_run_id},
            {
                "$set": {
                    "cancel_requested": True,
                    "cancel_requested_at": now,
                    "status": "cancelling",
                    "stage": "cancelling",
                    "updated_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        self.add_event(ingest_run_id, "cancel_requested", {})
        return self._enrich_run_status(updated)

    def is_cancel_requested(self, ingest_run_id: str) -> bool:
        run = self.runs.find_one({"ingest_run_id": ingest_run_id}, {"cancel_requested": 1})
        return bool(run and run.get("cancel_requested"))

    def mark_run_cancelled(self, ingest_run_id: str, lease_owner: str, reason: str) -> dict[str, Any] | None:
        now = _utcnow()
        run = self.runs.find_one_and_update(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {
                "$set": {
                    "status": "cancelled",
                    "stage": "finished",
                    "error": reason,
                    "finished_at": now,
                    "updated_at": now,
                },
                "$unset": {"lease_expires_at": "", "lease_owner": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
        if run is not None:
            self.add_event(ingest_run_id, "run_cancelled", {"reason": reason})
        return run

    def get_or_create_batch(self, ingest_run_id: str, start_line: int, end_line: int) -> dict[str, Any]:
        now = _utcnow()
        self.batches.update_one(
            {"ingest_run_id": ingest_run_id, "start_line": start_line, "end_line": end_line},
            {
                "$setOnInsert": {
                    "ingest_run_id": ingest_run_id,
                    "batch_id": uuid4().hex,
                    "start_line": start_line,
                    "end_line": end_line,
                    "status": "pending",
                    "attempt_count": 0,
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
        )
        return self.batches.find_one(
            {"ingest_run_id": ingest_run_id, "start_line": start_line, "end_line": end_line}
        )

    def mark_batch_running(self, batch_id: str) -> None:
        self.batches.update_one(
            {"batch_id": batch_id},
            {"$set": {"status": "running", "updated_at": _utcnow()}, "$inc": {"attempt_count": 1}},
        )

    def mark_batch_succeeded(
        self,
        batch_id: str,
        raw_count: int,
        cleaned_count: int,
        document_count: int,
        indexed_count: int,
        existing_count: int,
    ) -> None:
        self.batches.update_one(
            {"batch_id": batch_id},
            {
                "$set": {
                    "status": "succeeded",
                    "raw_count": raw_count,
                    "cleaned_count": cleaned_count,
                    "document_count": document_count,
                    "indexed_count": indexed_count,
                    "existing_count": existing_count,
                    "error": None,
                    "updated_at": _utcnow(),
                }
            },
        )

    def mark_batch_failed(
        self,
        batch_id: str,
        error: str,
        *,
        raw_count: int = 0,
        cleaned_count: int = 0,
        document_count: int = 0,
        indexed_count: int = 0,
    ) -> None:
        self.batches.update_one(
            {"batch_id": batch_id},
            {
                "$set": {
                    "status": "failed",
                    "raw_count": raw_count,
                    "cleaned_count": cleaned_count,
                    "document_count": document_count,
                    "indexed_count": indexed_count,
                    "error": error,
                    "updated_at": _utcnow(),
                }
            },
        )

    def update_run_progress(
        self,
        ingest_run_id: str,
        lease_owner: str,
        *,
        last_scanned_line: int,
        processed_batches_inc: int = 1,
        skipped_succeeded_batches_inc: int = 0,
        embedding_seconds_total_inc: float = 0.0,
        embedding_seconds_last_batch: float = 0.0,
    ) -> dict[str, Any]:
        return self.runs.find_one_and_update(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {
                "$max": {"last_scanned_line": last_scanned_line},
                "$inc": {
                    "processed_batches": processed_batches_inc,
                    "skipped_succeeded_batches": skipped_succeeded_batches_inc,
                    "embedding_seconds_total": float(embedding_seconds_total_inc),
                },
                "$set": {
                    "embedding_seconds_last_batch": float(embedding_seconds_last_batch),
                    "updated_at": _utcnow(),
                },
            },
            return_document=ReturnDocument.AFTER,
        )

    def update_last_scanned_line(self, ingest_run_id: str, last_scanned_line: int) -> None:
        self.runs.update_one(
            {"ingest_run_id": ingest_run_id},
            {"$max": {"last_scanned_line": last_scanned_line}, "$set": {"updated_at": _utcnow()}},
        )

    def finalize_run(self, ingest_run_id: str, lease_owner: str) -> dict[str, Any]:
        summary = self._summarize(ingest_run_id)
        status = "completed_with_errors" if summary["failed_batches"] else "completed"
        now = _utcnow()
        self.runs.update_one(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {
                "$set": {
                    **summary,
                    "status": status,
                    "stage": "finished",
                    "finished_at": now,
                    "lease_owner": None,
                    "updated_at": now,
                },
                "$unset": {"lease_expires_at": ""},
            },
        )
        self.add_event(ingest_run_id, "run_finished", {"status": status, **summary})
        return self.get_status(ingest_run_id)

    def fail_run(self, ingest_run_id: str, lease_owner: str, error: str) -> None:
        now = _utcnow()
        self.runs.update_one(
            {"ingest_run_id": ingest_run_id, "lease_owner": lease_owner},
            {
                "$set": {
                    "status": "failed",
                    "stage": "finished",
                    "error": error,
                    "finished_at": now,
                    "lease_owner": None,
                    "updated_at": now,
                },
                "$unset": {"lease_expires_at": ""},
            },
        )
        self.add_event(ingest_run_id, "run_failed", {"error": error})

    def _summarize(self, ingest_run_id: str) -> dict[str, int]:
        batches = list(self.batches.find({"ingest_run_id": ingest_run_id}).sort("start_line", ASCENDING))
        checkpoint_line = 0
        gap_found = False
        for batch in batches:
            if gap_found or batch.get("status") != "succeeded":
                gap_found = True
                continue
            checkpoint_line = max(checkpoint_line, int(batch["end_line"]))
        return {
            "raw_count": sum(int(batch.get("raw_count", 0)) for batch in batches),
            "cleaned_count": sum(int(batch.get("cleaned_count", 0)) for batch in batches),
            "document_count": sum(int(batch.get("document_count", 0)) for batch in batches),
            "indexed_count": sum(int(batch.get("indexed_count", 0)) for batch in batches),
            "succeeded_batches": sum(batch.get("status") == "succeeded" for batch in batches),
            "pending_batches": sum(batch.get("status") in {"pending", "running"} for batch in batches),
            "failed_batches": sum(batch.get("status") == "failed" for batch in batches),
            "checkpoint_line": checkpoint_line,
        }

    def _enrich_run_status(self, run: dict[str, Any]) -> dict[str, Any]:
        summary = self._summarize(run["ingest_run_id"])
        run.update(summary)
        run["current_scanned_line"] = int(run.get("last_scanned_line", 0) or 0)
        total_lines = int(run.get("total_lines", 0) or 0)
        run["cancel_requested"] = bool(run.get("cancel_requested", False))
        run["processed_batches"] = int(run.get("processed_batches", 0) or 0)
        run["skipped_succeeded_batches"] = int(run.get("skipped_succeeded_batches", 0) or 0)
        run["embedding_seconds_total"] = float(run.get("embedding_seconds_total", 0.0) or 0.0)
        run["embedding_seconds_last_batch"] = float(run.get("embedding_seconds_last_batch", 0.0) or 0.0)
        run["execution_outcome"] = self._get_execution_outcome(
            processed_batches=run["processed_batches"],
            skipped_succeeded_batches=run["skipped_succeeded_batches"],
        )
        run["stage"] = run.get("stage") or run.get("status")
        run["total_lines"] = total_lines

        started_at = run.get("started_at")
        finished_at = run.get("finished_at")
        end_time = finished_at or _utcnow()
        elapsed_seconds: float | None = None
        if started_at is not None:
            elapsed_seconds = max((end_time - started_at).total_seconds(), 0.0)
        run["elapsed_seconds"] = elapsed_seconds

        lines_per_second: float | None = None
        batches_per_second: float | None = None
        eta_seconds: float | None = None
        if elapsed_seconds and elapsed_seconds > 0:
            lines_per_second = run["current_scanned_line"] / elapsed_seconds
            batches_per_second = run["processed_batches"] / elapsed_seconds
            if total_lines > 0 and lines_per_second > 0:
                eta_seconds = max((total_lines - run["current_scanned_line"]) / lines_per_second, 0.0)
        run["lines_per_second"] = lines_per_second
        run["batches_per_second"] = batches_per_second
        run["eta_seconds"] = eta_seconds
        run["progress_percent"] = (
            (run["current_scanned_line"] / total_lines) * 100.0 if total_lines > 0 else 0.0
        )
        return run

    @staticmethod
    def _get_execution_outcome(*, processed_batches: int, skipped_succeeded_batches: int) -> str:
        if processed_batches <= 0:
            return "not_started"
        if processed_batches == skipped_succeeded_batches:
            return "skipped_all"
        return "processed"

    def get_status(self, ingest_run_id: str) -> dict[str, Any] | None:
        self.ping()
        run = self.runs.find_one({"ingest_run_id": ingest_run_id}, {"_id": 0})
        if run is None:
            return None
        failed = list(
            self.batches.find(
                {"ingest_run_id": ingest_run_id, "status": "failed"},
                {"_id": 0, "batch_id": 1, "start_line": 1, "end_line": 1, "attempt_count": 1, "error": 1},
            ).sort("start_line", ASCENDING)
        )
        run["failed_batch_details"] = failed
        return self._enrich_run_status(run)

    def add_event(self, ingest_run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.events.insert_one(
            {
                "ingest_run_id": ingest_run_id,
                "event_type": event_type,
                "payload": payload,
                "created_at": _utcnow(),
            }
        )
