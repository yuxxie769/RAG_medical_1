from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

from app.core.logging import get_logger
from app.data import clean_records, iter_raw_records
from app.kb import build_documents
from app.retrieval import VectorIndexer

from .repository import MongoIngestRepository


logger = get_logger(__name__)

INFRA_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
LINE_COUNT_HEARTBEAT_INTERVAL = 1000


class InfrastructureCircuitBreakerOpen(RuntimeError):
    """Raised after repeated infrastructure failures stop an ingest run."""


MilvusCircuitBreakerOpen = InfrastructureCircuitBreakerOpen


class IngestCancelled(RuntimeError):
    """Raised when an ingest run is cancelled cooperatively."""


@dataclass
class PreparedIngestRun:
    ingest_run_id: str
    lease_owner: str
    source_path: str
    raw_batch_id: str | None
    clean_batch_id: str | None
    kb_batch_id: str | None
    flush_to_milvus: bool
    force_reingest: bool
    target_collection_name: str | None


@dataclass
class BatchProcessingResult:
    outcome: Literal["succeeded", "skipped", "failed", "infra_failed"]
    batch_id: str
    start_line: int
    end_line: int
    attempt_count: int
    raw_count: int
    cleaned_count: int
    document_count: int
    indexed_count: int
    existing_count: int
    embedding_seconds: float
    batch_seconds: float
    doc_ids: tuple[str, ...] = ()
    failure_stage: Literal["embedding", "milvus"] | None = None
    error: str | None = None


class IngestService:
    def __init__(
        self,
        repository: MongoIngestRepository,
        indexer_factory: Callable[[], VectorIndexer],
        batch_size: int,
    ):
        self.repository = repository
        self.indexer_factory = indexer_factory
        self.batch_size = batch_size

    def ingest(
        self,
        source_path: str,
        raw_batch_id: str | None = None,
        clean_batch_id: str | None = None,
        kb_batch_id: str | None = None,
        flush_to_milvus: bool = True,
        force_reingest: bool = False,
        target_collection_name: str | None = None,
    ) -> dict[str, Any]:
        prepared = self.prepare_ingest(
            source_path=source_path,
            raw_batch_id=raw_batch_id,
            clean_batch_id=clean_batch_id,
            kb_batch_id=kb_batch_id,
            flush_to_milvus=flush_to_milvus,
            force_reingest=force_reingest,
            target_collection_name=target_collection_name,
        )
        return self.execute_prepared_ingest(prepared)

    def prepare_ingest(
        self,
        source_path: str,
        raw_batch_id: str | None = None,
        clean_batch_id: str | None = None,
        kb_batch_id: str | None = None,
        flush_to_milvus: bool = True,
        force_reingest: bool = False,
        target_collection_name: str | None = None,
    ) -> PreparedIngestRun:
        normalized_path = str(Path(source_path).resolve())
        file_path = Path(normalized_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        parameters = {
            "raw_batch_id": raw_batch_id,
            "clean_batch_id": clean_batch_id,
            "kb_batch_id": kb_batch_id,
            "flush_to_milvus": flush_to_milvus,
            "force_reingest": force_reingest,
            "target_collection_name": target_collection_name,
        }
        run = self.repository.start_run(
            normalized_path,
            target_collection_name,
            parameters,
            force_reingest,
        )
        logger.info(
            "Queued ingest run_id=%s source_path=%s collection=%s force_reingest=%s",
            run["ingest_run_id"],
            normalized_path,
            target_collection_name or "default",
            force_reingest,
        )
        return PreparedIngestRun(
            ingest_run_id=run["ingest_run_id"],
            lease_owner=run["lease_owner"],
            source_path=normalized_path,
            raw_batch_id=raw_batch_id,
            clean_batch_id=clean_batch_id,
            kb_batch_id=kb_batch_id,
            flush_to_milvus=flush_to_milvus,
            force_reingest=force_reingest,
            target_collection_name=target_collection_name,
        )

    def execute_prepared_ingest(self, prepared: PreparedIngestRun) -> dict[str, Any]:
        ingest_run_id = prepared.ingest_run_id
        lease_owner = prepared.lease_owner
        try:
            if self.repository.is_cancel_requested(ingest_run_id):
                return self._cancel_run(ingest_run_id, lease_owner, "Cancelled before ingest execution started")

            self.repository.mark_run_running(ingest_run_id, lease_owner, stage="counting_lines")
            logger.info("Started ingest run_id=%s source_path=%s", ingest_run_id, prepared.source_path)

            total_lines = self._count_total_lines(prepared.source_path, ingest_run_id, lease_owner)
            if self.repository.is_cancel_requested(ingest_run_id):
                return self._cancel_run(ingest_run_id, lease_owner, "Cancelled while counting source lines")

            self.repository.update_run_stage(ingest_run_id, lease_owner, "ingesting", total_lines=total_lines)
            indexer = self.indexer_factory()
            buffer: list[dict[str, Any]] = []
            consecutive_infra_failures = 0
            successful_write_results: list[BatchProcessingResult] = []
            for item in iter_raw_records(prepared.source_path, batch_id=prepared.raw_batch_id):
                if self.repository.is_cancel_requested(ingest_run_id):
                    raise IngestCancelled("Cancelled before processing the next batch")
                buffer.append(item)
                if len(buffer) >= self.batch_size:
                    consecutive_infra_failures, result = self._process_and_check_circuit_breaker(
                        ingest_run_id,
                        lease_owner,
                        buffer,
                        indexer,
                        prepared.clean_batch_id,
                        prepared.kb_batch_id,
                        prepared.flush_to_milvus,
                        consecutive_infra_failures,
                    )
                    if result.outcome == "succeeded" and result.indexed_count > 0:
                        successful_write_results.append(result)
                    buffer = []
            if buffer:
                if self.repository.is_cancel_requested(ingest_run_id):
                    raise IngestCancelled("Cancelled before processing the tail batch")
                _, result = self._process_and_check_circuit_breaker(
                    ingest_run_id,
                    lease_owner,
                    buffer,
                    indexer,
                    prepared.clean_batch_id,
                    prepared.kb_batch_id,
                    prepared.flush_to_milvus,
                    consecutive_infra_failures,
                )
                if result.outcome == "succeeded" and result.indexed_count > 0:
                    successful_write_results.append(result)
            if prepared.flush_to_milvus:
                self._flush_indexed_batches(ingest_run_id, lease_owner, indexer, successful_write_results)
            state = self.repository.finalize_run(ingest_run_id, lease_owner)
            logger.info("Finished ingest run_id=%s status=%s", ingest_run_id, state["status"])
            return state
        except IngestCancelled as exc:
            return self._cancel_run(ingest_run_id, lease_owner, str(exc))
        except Exception as exc:
            self.repository.fail_run(ingest_run_id, lease_owner, str(exc))
            logger.exception("Ingest run failed run_id=%s error=%s", ingest_run_id, exc)
            raise

    def _count_total_lines(self, source_path: str, ingest_run_id: str, lease_owner: str) -> int:
        total_lines = 0
        with Path(source_path).open("r", encoding="utf-8") as file_handle:
            for total_lines, _ in enumerate(file_handle, start=1):
                if total_lines % LINE_COUNT_HEARTBEAT_INTERVAL == 0:
                    self.repository.renew_lease(ingest_run_id, lease_owner)
                    if self.repository.is_cancel_requested(ingest_run_id):
                        raise IngestCancelled("Cancelled while counting source lines")
        return total_lines

    def _process_and_check_circuit_breaker(
        self,
        ingest_run_id: str,
        lease_owner: str,
        raw_records: list[dict[str, Any]],
        indexer: VectorIndexer,
        clean_batch_id: str | None,
        kb_batch_id: str | None,
        flush_to_milvus: bool,
        consecutive_infra_failures: int,
    ) -> tuple[int, BatchProcessingResult]:
        result = self._process_batch(
            ingest_run_id,
            lease_owner,
            raw_records,
            indexer,
            clean_batch_id,
            kb_batch_id,
            flush_to_milvus,
        )
        if result.outcome != "infra_failed":
            return 0, result

        consecutive_infra_failures += 1
        if consecutive_infra_failures < INFRA_CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            return consecutive_infra_failures, result

        error = (
            f"Infrastructure circuit breaker opened after {consecutive_infra_failures} "
            f"consecutive failures; stage={result.failure_stage}; "
            f"last batch={result.start_line}-{result.end_line}; "
            f"error={result.error}"
        )
        self.repository.add_event(
            ingest_run_id,
            "circuit_breaker_opened",
            {
                "failure_threshold": INFRA_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                "consecutive_failures": consecutive_infra_failures,
                "failure_stage": result.failure_stage,
                "batch_id": result.batch_id,
                "start_line": result.start_line,
                "end_line": result.end_line,
                "error": result.error,
            },
        )
        logger.error(
            "Opened infrastructure circuit breaker run_id=%s batch_id=%s lines=%s-%s stage=%s error=%s",
            ingest_run_id,
            result.batch_id,
            result.start_line,
            result.end_line,
            result.failure_stage,
            result.error,
        )
        raise InfrastructureCircuitBreakerOpen(error)

    def _flush_indexed_batches(
        self,
        ingest_run_id: str,
        lease_owner: str,
        indexer: VectorIndexer,
        results: list[BatchProcessingResult],
    ) -> None:
        if not results:
            return

        doc_ids = sorted({doc_id for result in results for doc_id in result.doc_ids})
        if not doc_ids:
            return

        try:
            self.repository.renew_lease(ingest_run_id, lease_owner)
            indexer.flush()
            indexer.verify_persisted_doc_ids(doc_ids)
        except Exception as exc:
            error = f"Deferred Milvus flush failed: {exc}"
            for result in results:
                self.repository.mark_batch_failed(
                    result.batch_id,
                    error,
                    raw_count=result.raw_count,
                    cleaned_count=result.cleaned_count,
                    document_count=result.document_count,
                    indexed_count=result.indexed_count,
                )
                self.repository.add_event(
                    ingest_run_id,
                    "batch_failed",
                    {
                        "batch_id": result.batch_id,
                        "start_line": result.start_line,
                        "end_line": result.end_line,
                        "error": error,
                        "deferred_flush_failure": True,
                    },
                )
            raise RuntimeError(error) from exc

    def _process_batch(
        self,
        ingest_run_id: str,
        lease_owner: str,
        raw_records: list[dict[str, Any]],
        indexer: VectorIndexer,
        clean_batch_id: str | None,
        kb_batch_id: str | None,
        flush_to_milvus: bool,
    ) -> BatchProcessingResult:
        batch_started = perf_counter()
        start_line = int(raw_records[0]["source_line_no"])
        end_line = int(raw_records[-1]["source_line_no"])
        batch = self.repository.get_or_create_batch(ingest_run_id, start_line, end_line)
        self.repository.renew_lease(ingest_run_id, lease_owner)
        if batch["status"] == "succeeded":
            result = BatchProcessingResult(
                outcome="skipped",
                batch_id=batch["batch_id"],
                start_line=start_line,
                end_line=end_line,
                attempt_count=int(batch.get("attempt_count", 0)),
                raw_count=len(raw_records),
                cleaned_count=0,
                document_count=0,
                indexed_count=0,
                existing_count=0,
                embedding_seconds=0.0,
                batch_seconds=perf_counter() - batch_started,
            )
            self._record_batch_progress(ingest_run_id, lease_owner, result)
            self.repository.add_event(
                ingest_run_id,
                "batch_skipped",
                {"batch_id": batch["batch_id"], "start_line": start_line, "end_line": end_line},
            )
            self._log_batch_result(ingest_run_id, result)
            return result

        batch_id = batch["batch_id"]
        attempt_count = int(batch.get("attempt_count", 0)) + 1
        self.repository.renew_lease(ingest_run_id, lease_owner)
        self.repository.mark_batch_running(batch_id)
        cleaned_records: list[dict[str, Any]] = []
        documents = []
        doc_ids: tuple[str, ...] = ()
        indexed_count = 0
        existing_count = 0
        embedding_seconds = 0.0
        failure_stage: Literal["embedding", "milvus"] | None = None
        try:
            cleaned_records = clean_records(raw_records, clean_batch_id=clean_batch_id)
            documents = build_documents(cleaned_records, batch_id=kb_batch_id)
            doc_ids = tuple(document.doc_id for document in documents)
            if flush_to_milvus and documents:
                indexing_result = indexer.index_in_batches(documents)
                indexed_count = indexing_result.indexed_count
                existing_count = indexing_result.existing_count
                embedding_seconds = indexing_result.embedding_seconds
                failure_stage = indexing_result.failure_stage
                if indexing_result.failed_count or indexing_result.errors:
                    raise RuntimeError("; ".join(indexing_result.errors) or "Milvus indexing failed")
                if indexing_result.indexed_count + indexing_result.existing_count != len(documents):
                    failure_stage = "milvus"
                    raise RuntimeError("Milvus indexing did not account for every document")
            self.repository.mark_batch_succeeded(
                batch_id=batch_id,
                raw_count=len(raw_records),
                cleaned_count=len(cleaned_records),
                document_count=len(documents),
                indexed_count=indexed_count,
                existing_count=existing_count,
            )
            result = BatchProcessingResult(
                outcome="succeeded",
                batch_id=batch_id,
                start_line=start_line,
                end_line=end_line,
                attempt_count=attempt_count,
                raw_count=len(raw_records),
                cleaned_count=len(cleaned_records),
                document_count=len(documents),
                indexed_count=indexed_count,
                existing_count=existing_count,
                embedding_seconds=embedding_seconds,
                batch_seconds=perf_counter() - batch_started,
                doc_ids=doc_ids,
            )
            self._record_batch_progress(ingest_run_id, lease_owner, result)
            self.repository.add_event(
                ingest_run_id,
                "batch_succeeded",
                {"batch_id": batch_id, "start_line": start_line, "end_line": end_line},
            )
            self._log_batch_result(ingest_run_id, result)
            return result
        except Exception as exc:
            self.repository.mark_batch_failed(
                batch_id,
                str(exc),
                raw_count=len(raw_records),
                cleaned_count=len(cleaned_records),
                document_count=len(documents),
                indexed_count=indexed_count,
            )
            result = BatchProcessingResult(
                outcome="infra_failed" if failure_stage in {"embedding", "milvus"} else "failed",
                batch_id=batch_id,
                start_line=start_line,
                end_line=end_line,
                attempt_count=attempt_count,
                raw_count=len(raw_records),
                cleaned_count=len(cleaned_records),
                document_count=len(documents),
                indexed_count=indexed_count,
                existing_count=existing_count,
                embedding_seconds=embedding_seconds,
                batch_seconds=perf_counter() - batch_started,
                doc_ids=doc_ids,
                failure_stage=failure_stage,
                error=str(exc),
            )
            self._record_batch_progress(ingest_run_id, lease_owner, result)
            self.repository.add_event(
                ingest_run_id,
                "batch_failed",
                {
                    "batch_id": batch_id,
                    "start_line": start_line,
                    "end_line": end_line,
                    "failure_stage": failure_stage,
                    "error": str(exc),
                },
            )
            self._log_batch_result(ingest_run_id, result)
            return result

    def _record_batch_progress(
        self,
        ingest_run_id: str,
        lease_owner: str,
        result: BatchProcessingResult,
    ) -> dict[str, Any]:
        skipped_inc = 1 if result.outcome == "skipped" else 0
        return self.repository.update_run_progress(
            ingest_run_id,
            lease_owner,
            last_scanned_line=result.end_line,
            processed_batches_inc=1,
            skipped_succeeded_batches_inc=skipped_inc,
            embedding_seconds_total_inc=result.embedding_seconds,
            embedding_seconds_last_batch=result.embedding_seconds,
        )

    def _log_batch_result(self, ingest_run_id: str, result: BatchProcessingResult) -> None:
        run_state = self.repository.get_status(ingest_run_id)
        if run_state is None:
            return
        level = logger.error if result.outcome in {"failed", "infra_failed"} else logger.info
        level(
            "batch_%s run_id=%s batch_id=%s lines=%s-%s attempt=%s raw=%s cleaned=%s docs=%s indexed=%s "
            "existing=%s failure_stage=%s batch_seconds=%.3f embedding_seconds=%.3f processed_batches=%s "
            "lines_per_second=%s batches_per_second=%s eta_seconds=%s error=%s",
            result.outcome,
            ingest_run_id,
            result.batch_id,
            result.start_line,
            result.end_line,
            result.attempt_count,
            result.raw_count,
            result.cleaned_count,
            result.document_count,
            result.indexed_count,
            result.existing_count,
            result.failure_stage,
            result.batch_seconds,
            result.embedding_seconds,
            run_state.get("processed_batches", 0),
            self._format_metric(run_state.get("lines_per_second")),
            self._format_metric(run_state.get("batches_per_second")),
            self._format_metric(run_state.get("eta_seconds")),
            result.error,
        )

    def _cancel_run(self, ingest_run_id: str, lease_owner: str, reason: str) -> dict[str, Any]:
        cancelled_run = self.repository.mark_run_cancelled(ingest_run_id, lease_owner, reason)
        logger.info("Cancelled ingest run_id=%s reason=%s", ingest_run_id, reason)
        return self.repository.get_status(ingest_run_id) if cancelled_run is not None else {}

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"
