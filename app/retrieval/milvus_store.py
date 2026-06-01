from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from app.core.models import Document, RetrievalResult


SearchMethod = Literal["hybrid", "dense", "sparse"]


@dataclass
class MilvusRecord:
    doc_id: str
    vector: List[float]
    content: str
    question: str
    answer: str
    metadata: Dict[str, Any]


@dataclass
class MilvusCollectionSchema:
    collection_name: str
    vector_field_name: str = "embedding"
    sparse_vector_field_name: str = "sparse_embedding"
    primary_field_name: str = "doc_id"
    text_field_name: str = "content"
    dimension: int = 1024
    dense_metric_type: str = "COSINE"
    analyzer_type: str = "chinese"
    rrf_k: int = 60
    metadata_fields: List[str] = field(default_factory=lambda: [
        "question",
        "answer",
        "source",
        "split",
        "clean_batch_id",
        "raw_batch_id",
        "source_path",
        "source_line_no",
        "kb_batch_id",
        "kb_index",
    ])


class MilvusStore:
    def __init__(
        self,
        schema: MilvusCollectionSchema,
        host: str = "localhost",
        port: int = 19530,
    ):
        self.schema = schema
        self.host = host
        self.port = port
        self.client: MilvusClient | None = None

    def connect(self) -> MilvusClient:
        if self.client is None:
            self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        return self.client

    def _build_schema(self):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=self.schema.primary_field_name,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        schema.add_field(
            field_name=self.schema.vector_field_name,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.schema.dimension,
        )
        schema.add_field(
            field_name=self.schema.sparse_vector_field_name,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name=self.schema.text_field_name,
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": self.schema.analyzer_type},
        )
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="split", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="clean_batch_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="raw_batch_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="source_line_no", datatype=DataType.INT64)
        schema.add_field(field_name="kb_batch_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="kb_index", datatype=DataType.INT64)
        schema.add_function(
            Function(
                name="content_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[self.schema.text_field_name],
                output_field_names=[self.schema.sparse_vector_field_name],
            )
        )
        return schema

    def _build_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=self.schema.vector_field_name,
            index_type="HNSW",
            metric_type=self.schema.dense_metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name=self.schema.sparse_vector_field_name,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        return index_params

    def _validate_existing_schema(self) -> None:
        client = self.connect()
        description = client.describe_collection(collection_name=self.schema.collection_name)
        field_names = {field["name"] for field in description.get("fields", [])}
        required_fields = {
            self.schema.primary_field_name,
            self.schema.vector_field_name,
            self.schema.sparse_vector_field_name,
            self.schema.text_field_name,
        }
        if not required_fields.issubset(field_names):
            raise RuntimeError(
                "Milvus collection uses the old dense-only schema. "
                "Run scripts/reset_milvus_collection.py --confirm-drop and ingest the data again."
            )

        functions = description.get("functions", [])
        has_bm25 = any(
            function.get("type") in {FunctionType.BM25, int(FunctionType.BM25)}
            or str(function.get("type", "")).upper() == "BM25"
            for function in functions
        )
        if not has_bm25:
            raise RuntimeError(
                "Milvus collection is missing the BM25 function. "
                "Run scripts/reset_milvus_collection.py --confirm-drop and ingest the data again."
            )

    def create_collection(self) -> None:
        client = self.connect()
        if client.has_collection(collection_name=self.schema.collection_name):
            self._validate_existing_schema()
            client.load_collection(collection_name=self.schema.collection_name)
            return

        client.create_collection(
            collection_name=self.schema.collection_name,
            schema=self._build_schema(),
            index_params=self._build_index_params(),
        )
        client.load_collection(collection_name=self.schema.collection_name)

    def _existing_doc_ids(self, doc_ids: Sequence[str]) -> set[str]:
        if not doc_ids:
            return set()
        self.create_collection()
        rows = self.connect().query(
            collection_name=self.schema.collection_name,
            filter=f'{self.schema.primary_field_name} in [{", ".join(repr(doc_id) for doc_id in doc_ids)}]',
            output_fields=[self.schema.primary_field_name],
        )
        return {str(row[self.schema.primary_field_name]) for row in rows}

    def upsert(self, records: Sequence[MilvusRecord]) -> None:
        if not records:
            return
        self.create_collection()
        existing_doc_ids = self._existing_doc_ids([record.doc_id for record in records])
        if existing_doc_ids:
            raise ValueError(f"Duplicate doc_id exists in Milvus: {sorted(existing_doc_ids)}")

        rows = [
            {
                self.schema.primary_field_name: record.doc_id,
                self.schema.vector_field_name: record.vector,
                self.schema.text_field_name: record.content,
                "question": record.question,
                "answer": record.answer,
                "source": str(record.metadata.get("source", "unknown")),
                "split": str(record.metadata.get("split", "unknown")),
                "clean_batch_id": str(record.metadata.get("clean_batch_id", "unknown")),
                "raw_batch_id": str(record.metadata.get("raw_batch_id", "unknown")),
                "source_path": str(record.metadata.get("source_path", "unknown")),
                "source_line_no": int(record.metadata.get("source_line_no") or 0),
                "kb_batch_id": str(record.metadata.get("kb_batch_id", "unknown")),
                "kb_index": int(record.metadata.get("kb_index") or 0),
            }
            for record in records
        ]
        client = self.connect()
        client.insert(collection_name=self.schema.collection_name, data=rows)
        client.flush(collection_name=self.schema.collection_name)

    def search(
        self,
        query_vector: List[float] | None = None,
        query_text: str | None = None,
        limit: int = 5,
        search_method: SearchMethod = "hybrid",
    ) -> List[RetrievalResult]:
        self.create_collection()
        client = self.connect()
        output_fields = [
            self.schema.primary_field_name,
            self.schema.text_field_name,
            *self.schema.metadata_fields,
        ]

        if search_method == "dense":
            if query_vector is None:
                raise ValueError("Dense search requires a query vector.")
            results = client.search(
                collection_name=self.schema.collection_name,
                data=[query_vector],
                anns_field=self.schema.vector_field_name,
                search_params={"metric_type": self.schema.dense_metric_type, "params": {"ef": 64}},
                limit=limit,
                output_fields=output_fields,
            )
        elif search_method == "sparse":
            if not query_text:
                raise ValueError("Sparse search requires query text.")
            results = client.search(
                collection_name=self.schema.collection_name,
                data=[query_text],
                anns_field=self.schema.sparse_vector_field_name,
                search_params={"metric_type": "BM25", "params": {}},
                limit=limit,
                output_fields=output_fields,
            )
        elif search_method == "hybrid":
            if query_vector is None or not query_text:
                raise ValueError("Hybrid search requires both a query vector and query text.")
            results = client.hybrid_search(
                collection_name=self.schema.collection_name,
                reqs=[
                    AnnSearchRequest(
                        data=[query_vector],
                        anns_field=self.schema.vector_field_name,
                        param={"metric_type": self.schema.dense_metric_type, "params": {"ef": 64}},
                        limit=limit,
                    ),
                    AnnSearchRequest(
                        data=[query_text],
                        anns_field=self.schema.sparse_vector_field_name,
                        param={"metric_type": "BM25", "params": {}},
                        limit=limit,
                    ),
                ],
                ranker=RRFRanker(k=self.schema.rrf_k),
                limit=limit,
                output_fields=output_fields,
            )
        else:
            raise ValueError(f"Unsupported search method: {search_method}")

        return self._to_retrieval_results(results)

    def _to_retrieval_results(self, results: List[List[dict]]) -> List[RetrievalResult]:
        hits: List[RetrievalResult] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {})
            doc_id = hit.get(self.schema.primary_field_name) or hit.get("id") or getattr(hit, "pk", "")
            metadata = {
                field: entity.get(field)
                for field in self.schema.metadata_fields
                if field in entity
            }
            hits.append(
                RetrievalResult(
                    doc_id=str(doc_id),
                    content=str(entity.get(self.schema.text_field_name, "")),
                    score=float(hit.get("distance", hit.get("score", 0.0))),
                    metadata=metadata,
                )
            )
        return hits

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return
        self.create_collection()
        client = self.connect()
        client.delete(collection_name=self.schema.collection_name, ids=list(doc_ids))
        client.flush(collection_name=self.schema.collection_name)

    def count(self) -> int:
        self.create_collection()
        stats = self.connect().get_collection_stats(collection_name=self.schema.collection_name)
        return int(stats.get("row_count", 0))


@dataclass
class InMemoryMilvusStore(MilvusStore):
    """Temporary in-memory store for local development and tests."""

    _records: Dict[str, MilvusRecord] = field(default_factory=dict)

    def __init__(self, schema: MilvusCollectionSchema, host: str = "localhost", port: int = 19530):
        super().__init__(schema, host=host, port=port)
        self._records = {}

    def create_collection(self) -> None:
        return None

    def upsert(self, records: Sequence[MilvusRecord]) -> None:
        for record in records:
            if record.doc_id in self._records:
                raise ValueError(f"Duplicate doc_id exists in memory store: {record.doc_id}")
            self._records[record.doc_id] = record

    def search(
        self,
        query_vector: List[float] | None = None,
        query_text: str | None = None,
        limit: int = 5,
        search_method: SearchMethod = "hybrid",
    ) -> List[RetrievalResult]:
        scored: List[RetrievalResult] = []
        query_sum = sum(query_vector or [])
        query_terms = set((query_text or "").lower().split())
        for record in self._records.values():
            dense_score = 1.0 / (1.0 + abs(sum(record.vector) - query_sum))
            sparse_score = float(len(query_terms & set(record.content.lower().split())))
            score = sparse_score if search_method == "sparse" else dense_score
            if search_method == "hybrid":
                score += sparse_score
            scored.append(
                RetrievalResult(
                    doc_id=record.doc_id,
                    content=record.content,
                    score=score,
                    metadata=record.metadata,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def delete_by_doc_ids(self, doc_ids: Sequence[str]) -> None:
        for doc_id in doc_ids:
            self._records.pop(doc_id, None)

    def count(self) -> int:
        return len(self._records)


def build_records(documents: Sequence[Document], vectors: Sequence[List[float]]) -> List[MilvusRecord]:
    records: List[MilvusRecord] = []
    for doc, vector in zip(documents, vectors):
        records.append(
            MilvusRecord(
                doc_id=doc.doc_id,
                vector=vector,
                content=doc.content,
                question=doc.question,
                answer=doc.answer,
                metadata=doc.metadata,
            )
        )
    return records
